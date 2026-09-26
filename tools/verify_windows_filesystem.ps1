# Native acceptance only: never run this account/security setup on a developer PC.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $Python,
    [switch] $Child,
    [string] $ExpectedSid,
    [switch] $Diagnostic,
    [string] $OutputDirectory = "ci-output"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $IsWindows -or $env:GITHUB_ACTIONS -ne 'true' -or
    $env:RUNNER_ENVIRONMENT -ne 'github-hosted') {
    throw 'This acceptance script requires an ephemeral GitHub-hosted Windows VM.'
}

if ($Child) {
    $Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    if ($Identity.User.Value -ne $ExpectedSid -or
        'S-1-5-32-544' -in @($Identity.Groups.Value)) {
        throw 'The test process must have the expected standard-user token.'
    }
    [ordered]@{
        user = $Identity.Name
        sid = $Identity.User.Value
        groups = @($Identity.Groups.Value)
        source = (Get-Location).Path
        profile = $env:USERPROFILE
        temporary = [IO.Path]::GetTempPath()
        python = $Python
    } | ConvertTo-Json -Depth 3 | Write-Output
    $Probe = Join-Path (Get-Location) ('.write-probe-' + [guid]::NewGuid().ToString('N'))
    try {
        $Stream = [IO.File]::Open($Probe, [IO.FileMode]::CreateNew)
        $Stream.Dispose()
        Remove-Item -LiteralPath $Probe
        throw 'The installation unexpectedly permits standard-user writes.'
    } catch [UnauthorizedAccessException] {
        Write-Output 'Confirmed: the installation denies standard-user writes.'
    }
    if ($Diagnostic -and $env:RAYCHAT_DIAGNOSTIC_TESTS) {
        $Tests = $env:RAYCHAT_DIAGNOSTIC_TESTS.Split(' ', [StringSplitOptions]::RemoveEmptyEntries)
        & $Python -B -S -X faulthandler -m unittest -v -f @Tests
    } else {
        & $Python -B -m tools.ci_release --output $OutputDirectory
    }
    exit $LASTEXITCODE
}

$Reports = New-Item -ItemType Directory -Force 'ci-output/windows-standard-user'
$Reports = $Reports.FullName
$TestRoot = Join-Path "$env:SystemDrive\" ('raychat-fs-' + [guid]::NewGuid().ToString('N'))
$AccountName = 'raychat_' + [guid]::NewGuid().ToString('N').Substring(0, 10)
$Account = $null
$Process = $null
$Failure = $null
$Retired = $true
$ChildOutput = $null
$Password = ConvertTo-SecureString ('Rc!9' + [guid]::NewGuid().ToString('N')) -AsPlainText -Force

function Save-DefenderState([string] $Name) {
    $Status = Get-MpComputerStatus | Select-Object AMRunningMode, AMServiceEnabled,
        AMProductVersion, AntivirusEnabled, AntivirusSignatureVersion,
        AntivirusSignatureLastUpdated, RealTimeProtectionEnabled, BehaviorMonitorEnabled,
        IoavProtectionEnabled, OnAccessProtectionEnabled, IsTamperProtected
    $Preferences = Get-MpPreference | Select-Object DisableRealtimeMonitoring,
        DisableBehaviorMonitoring, DisableIOAVProtection, DisableScriptScanning,
        DisableArchiveScanning, DisableAutoExclusions, RealTimeScanDirection,
        ExclusionPath, ExclusionProcess, ExclusionExtension
    [ordered]@{
        imageOS = $env:ImageOS
        imageVersion = $env:ImageVersion
        status = $Status
        preferences = $Preferences
    } | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $Reports "$Name.json")
    return @{ Status = $Status; Preferences = $Preferences }
}

function Assert-DefenderEnabled([string] $Name) {
    $State = Save-DefenderState $Name
    $Status = $State.Status
    $Preferences = $State.Preferences
    if (-not $Status.AMServiceEnabled -or -not $Status.AntivirusEnabled -or
        -not $Status.RealTimeProtectionEnabled -or -not $Status.BehaviorMonitorEnabled -or
        -not $Status.IoavProtectionEnabled -or -not $Status.OnAccessProtectionEnabled -or
        $Preferences.DisableRealtimeMonitoring -or $Preferences.DisableBehaviorMonitoring -or
        $Preferences.DisableIOAVProtection -or $Preferences.DisableScriptScanning -or
        $Preferences.DisableArchiveScanning -or -not $Preferences.DisableAutoExclusions -or
        $Preferences.RealTimeScanDirection -ne 0 -or
        @($Preferences.ExclusionPath).Where({ $_ }).Count -ne 0 -or
        @($Preferences.ExclusionProcess).Where({ $_ }).Count -ne 0 -or
        @($Preferences.ExclusionExtension).Where({ $_ }).Count -ne 0) {
        throw "Defender coverage is not enabled; inspect $Name.json."
    }
}

try {
    Write-Output "Provisioning ordinary-user acceptance at $([DateTime]::UtcNow.ToString('o'))."
    $Account = New-LocalUser -Name $AccountName -Password $Password -AccountNeverExpires
    Add-LocalGroupMember -SID 'S-1-5-32-545' -Member $Account
    $Credential = [pscredential]::new("$env:COMPUTERNAME\$AccountName", $Password)
    $null = New-Item -ItemType Directory $TestRoot
    $Acl = [Security.AccessControl.DirectorySecurity]::new()
    $Acl.SetAccessRuleProtection($true, $false)
    foreach ($Sid in @('S-1-5-18', 'S-1-5-32-544', $Account.SID.Value)) {
        $Rights = if ($Sid -eq $Account.SID.Value) { 'ReadAndExecute' } else { 'FullControl' }
        $Rule = [Security.AccessControl.FileSystemAccessRule]::new(
            [Security.Principal.SecurityIdentifier]::new($Sid), $Rights,
            'ContainerInherit,ObjectInherit', 'None', 'Allow')
        $Acl.AddAccessRule($Rule)
    }
    Set-Acl -LiteralPath $TestRoot -AclObject $Acl
    $Source = Join-Path $TestRoot 'source'
    $Archive = Join-Path $TestRoot 'source.zip'
    & git archive --format=zip --output $Archive HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Could not archive the tested commit.' }
    Write-Output "Extracting tested source at $([DateTime]::UtcNow.ToString('o'))."
    [IO.Compression.ZipFile]::ExtractToDirectory($Archive, $Source)
    $Data = New-Item -ItemType Directory (Join-Path $TestRoot 'data')
    $DataAcl = Get-Acl -LiteralPath $Data.FullName
    $DataAcl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new(
        $Account.SID, 'Modify', 'ContainerInherit,ObjectInherit', 'None', 'Allow'))
    Set-Acl -LiteralPath $Data.FullName -AclObject $DataAcl
    $ChildOutput = (New-Item -ItemType Directory (Join-Path $Data.FullName 'ci-output')).FullName
    $TestHome = (New-Item -ItemType Directory (Join-Path $Data.FullName 'profile')).FullName
    $TestTemp = (New-Item -ItemType Directory (Join-Path $Data.FullName 'temp')).FullName
    $Denied = (New-Item -ItemType Directory (Join-Path $TestRoot 'denied')).FullName
    [IO.File]::WriteAllBytes((Join-Path $Denied 'snapshot'), [Text.Encoding]::UTF8.GetBytes('old'))
    $OtherDrive = [IO.Path]::GetPathRoot($env:RUNNER_TEMP)
    if ($OtherDrive -eq [IO.Path]::GetPathRoot($TestRoot)) {
        throw 'Cross-volume acceptance requires a second local volume.'
    }
    $OtherRoot = Join-Path $OtherDrive ('raychat-fs-' + [guid]::NewGuid().ToString('N'))
    $null = New-Item -ItemType Directory $OtherRoot
    $DataAcl.SetAccessRuleProtection($true, $true)
    Set-Acl -LiteralPath $OtherRoot -AclObject $DataAcl
    $ChildEnvironment = @{
        USERPROFILE = $TestHome; TEMP = $TestTemp; TMP = $TestTemp
        APPDATA = $TestHome; LOCALAPPDATA = $TestHome
        PYTHONUTF8 = '1'; PYTHONIOENCODING = 'utf-8'; PYTHONDONTWRITEBYTECODE = '1'
        RAYCHAT_TEST_SID = $Account.SID.Value
        RAYCHAT_TEST_DENIED_DIRECTORY = $Denied; RAYCHAT_TEST_OTHER_VOLUME = $OtherRoot
    }
    $null = Save-DefenderState 'inherited'
    foreach ($Phase in @('defender')) {
        if ($Phase -eq 'defender') {
            Write-Output "Enabling Defender at $([DateTime]::UtcNow.ToString('o'))."
            # Strengthen this disposable VM's protection; never add exclusions.
            $Preferences = Get-MpPreference
            foreach ($Kind in @('ExclusionPath', 'ExclusionProcess', 'ExclusionExtension')) {
                $Values = @($Preferences.$Kind).Where({ $_ })
                if ($Values.Count) {
                    $Removal = @{}
                    $Removal[$Kind] = $Values
                    Remove-MpPreference @Removal
                }
            }
            $PassiveKey = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows Advanced Threat Protection'
            if (Test-Path -LiteralPath $PassiveKey) {
                Set-ItemProperty -LiteralPath $PassiveKey -Name ForceDefenderPassiveMode -Value 0
            }
            Start-Service WinDefend
            Set-MpPreference -DisableRealtimeMonitoring $false -DisableBehaviorMonitoring $false `
                -DisableIOAVProtection $false -DisableScriptScanning $false `
                -DisableArchiveScanning $false -DisableAutoExclusions $true -RealTimeScanDirection Both
            # Applying preferences is asynchronous. Observe activation for at
            # most thirty seconds, without replaying settings or running tests early.
            $Activation = [Diagnostics.Stopwatch]::StartNew()
            do {
                $Status = Get-MpComputerStatus
                if ($Status.RealTimeProtectionEnabled -and $Status.BehaviorMonitorEnabled -and
                    $Status.IoavProtectionEnabled -and $Status.OnAccessProtectionEnabled) { break }
                Start-Sleep -Milliseconds 500
            } while ($Activation.Elapsed.TotalSeconds -lt 30)
            Assert-DefenderEnabled 'enabled-before'
        }
        $Launch = @{
            FilePath = (Get-Process -Id $PID).Path
            ArgumentList = '-NoLogo -NoProfile -NonInteractive -File "' +
                (Join-Path $Source 'tools/verify_windows_filesystem.ps1') +
                '" -Child -Python "' + $Python + '" -ExpectedSid ' + $Account.SID.Value +
                ' -OutputDirectory "' + $ChildOutput + '"' + $(if ($Diagnostic) { ' -Diagnostic' } else { '' })
            Credential = $Credential
            LoadUserProfile = $true
            Environment = $ChildEnvironment
            WorkingDirectory = $Source
            RedirectStandardOutput = Join-Path $Reports "$Phase.stdout.log"
            RedirectStandardError = Join-Path $Reports "$Phase.stderr.log"
            PassThru = $true
        }
        Write-Output "Starting tests at $([DateTime]::UtcNow.ToString('o'))."
        $Process = Start-Process @Launch
        $Retired = $false
        # Short waits keep cancellation responsive and expose each Python stage.
        $Waiting = [Diagnostics.Stopwatch]::StartNew()
        $PrintedLines = @{}
        $BudgetMinutes = if ($Diagnostic -and $env:RAYCHAT_DIAGNOSTIC_TESTS) { 8 } else { 20 }
        do {
            $Exited = $Process.WaitForExit(1000)
            foreach ($LogPath in @($Launch.RedirectStandardOutput, $Launch.RedirectStandardError)) {
                $Lines = @(Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue)
                $Count = if ($PrintedLines.ContainsKey($LogPath)) { $PrintedLines[$LogPath] } else { 0 }
                if ($Lines.Count -gt $Count) {
                    $Lines[$Count..($Lines.Count - 1)] | Write-Output
                    $PrintedLines[$LogPath] = $Lines.Count
                }
            }
            if ($Waiting.Elapsed.TotalMinutes -ge $BudgetMinutes) {
                throw "$Phase tests exceeded $BudgetMinutes minutes."
            }
        } while (-not $Exited)
        $Retired = $true
        Get-Content -LiteralPath (Join-Path $Reports "$Phase.stderr.log")
        if ($Phase -eq 'defender') { Assert-DefenderEnabled 'enabled-after' }
        if ($Process.ExitCode -ne 0) { throw "$Phase tests exited $($Process.ExitCode)." }
        $Process.Dispose()
        $Process = $null
    }
} catch {
    $Failure = $_
} finally {
    try {
        if ($null -ne $Process) {
            if (-not $Retired) {
                $Process.Kill($true)
                $Retired = $Process.WaitForExit(30000)
                if (-not $Retired) { throw 'Test process did not retire after termination.' }
            }
            $Process.Dispose()
        }
        if ($Retired -and $null -ne $ChildOutput -and (Test-Path -LiteralPath $ChildOutput)) {
            Copy-Item -Path (Join-Path $ChildOutput "*") -Destination (Split-Path $Reports) -Recurse -Force
        }
        if ($null -ne $Account -and $Retired) { Remove-LocalUser -SID $Account.SID }
    } catch {
        if ($null -eq $Failure) { $Failure = $_ } else { Write-Warning $_ }
    }
    $Password.Dispose()
    # Retain this unique source/data tree until hosted-VM disposal. Never delete
    # a possibly active consumer's files, or mutate the repository/interpreter ACL.
    Write-Output "Acceptance scratch retained until VM disposal: $TestRoot"
}
if ($null -ne $Failure) { throw $Failure }
