# Native acceptance only: never run this account/security setup on a developer PC.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $Python,
    [switch] $Child,
    [switch] $IdleDiagnostic,
    [switch] $InspectHost,
    [string] $ExpectedSid,
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
        processors = [Environment]::ProcessorCount
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
    & $Python -B -m tools.ci_release --output $OutputDirectory --release-only
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    # Each discovered module runs once. Supervise native unittest processes
    # directly: a crashed interpreter must fail, not strand a process-pool task.
    $Modules = @(Get-ChildItem tests -Recurse -File -Filter 'test*.py' | Sort-Object FullName |
        ForEach-Object {
            ([IO.Path]::GetRelativePath((Get-Location).Path, $_.FullName) -replace '\.py$', '') -replace '[\\/]', '.'
        })
    if ($IdleDiagnostic) { $Modules = @('diagnostic_idle') }
    if ($Modules.Count -eq 0) { throw 'No unit-test modules were discovered.' }
    if ($IdleDiagnostic) {
        $ExpectedTests = 1
        $IdleScript = Join-Path $OutputDirectory 'diagnostic_idle.py'
        @'
import time
import unittest

class IdleLifetime(unittest.TestCase):
    def test_standard_user_remains_alive_for_twelve_minutes(self):
        for second in range(720):
            if second % 30 == 0:
                print(f"Idle diagnostic heartbeat: {second} seconds", flush=True)
            time.sleep(1)

unittest.main(verbosity=2)
'@ | Set-Content -Encoding utf8 $IdleScript
    } else {
        $ExpectedTests = & $Python -B -S -c "import unittest; print(unittest.defaultTestLoader.discover('tests').countTestCases())"
    }
    if ($LASTEXITCODE -ne 0) { throw 'Unit discovery failed.' }
    $ExpectedTests = [int]$ExpectedTests
    $Shards = @()
    $UnitClock = [Diagnostics.Stopwatch]::StartNew()
    $UnitExit = 1
    $PrintedUnitLines = @{}
    try {
        # Run the concurrency benchmark in a fresh process before other suites.
        # All discovered tests still run exactly once under the same deadline.
        $Phases = if ($IdleDiagnostic) { @('remaining') } else { @('stress', 'remaining') }
        foreach ($Phase in $Phases) {
            $PhaseModules = @(if ($Phase -eq 'stress') {
                'tests.test_workflow_stress'
            } else {
                $Modules | Where-Object { $_ -ne 'tests.test_workflow_stress' }
            })
            $Parallelism = if ($Phase -eq 'remaining') { 3 } else { 1 }
            $ActiveShards = @{}
            $NextModule = 0
            Write-Output "Running $Phase phase: $($PhaseModules.Count) modules in fresh interpreters, at most $Parallelism concurrently."
            do {
                $Launched = $false
                for ($GroupIndex = 0; $GroupIndex -lt $Parallelism; $GroupIndex++) {
                    if ($ActiveShards.ContainsKey($GroupIndex)) { continue }
                    if ($NextModule -ge $PhaseModules.Count) { break }
                    $Selection = @($PhaseModules[$NextModule])
                    $NextModule++
                    $Index = $Shards.Count
                    $UnitOut = Join-Path $OutputDirectory "unit-$Index.stdout.log"
                    $UnitErr = Join-Path $OutputDirectory "unit-$Index.stderr.log"
                    # Independent test processes must not contend for the application's
                    # default package store. Descendants within each module still share
                    # its profile, preserving the real cross-process ownership tests.
                    $ProfileIndex = if ($Phase -eq 'stress') { 0 } else { $GroupIndex + 1 }
                    $UnitProfile = (New-Item -ItemType Directory -Force (
                        Join-Path $env:USERPROFILE "unit-$ProfileIndex")).FullName
                    $UnitTemp = (New-Item -ItemType Directory -Force (
                        Join-Path $env:TEMP "unit-$ProfileIndex")).FullName
                    $UnitLaunch = @{
                        FilePath = $Python
                        ArgumentList = @('-B', '-S', '-X', 'faulthandler', '-m', 'unittest', '-v', '--durations', '20') + $Selection
                        WorkingDirectory = (Get-Location).Path
                        RedirectStandardOutput = $UnitOut
                        RedirectStandardError = $UnitErr
                        Environment = @{
                            HOME = $UnitProfile; USERPROFILE = $UnitProfile
                            APPDATA = $UnitProfile; LOCALAPPDATA = $UnitProfile
                            TEMP = $UnitTemp; TMP = $UnitTemp
                        }
                        PassThru = $true
                    }
                    if ($IdleDiagnostic) { $UnitLaunch.ArgumentList = @('-B', '-S', $IdleScript) }
                    $Shard = [pscustomobject]@{
                        Slot = $GroupIndex; Process = (Start-Process @UnitLaunch)
                        Stdout = $UnitOut; Stderr = $UnitErr; Modules = $Selection
                        Profile = $UnitProfile; Temporary = $UnitTemp
                    }
                    $Shards += $Shard
                    $ActiveShards[$GroupIndex] = $Shard
                    $Launched = $true
                }
                if ($Launched) {
                    $Shards | Select-Object Modules, Profile, Temporary | ConvertTo-Json -Depth 4 |
                        Set-Content -Encoding utf8 (Join-Path $OutputDirectory 'unit-modules.json')
                }
                foreach ($Shard in @($ActiveShards.Values)) {
                    $Exited = $Shard.Process.WaitForExit(250)
                    foreach ($UnitLog in @($Shard.Stdout, $Shard.Stderr)) {
                        $Lines = @(Get-Content -LiteralPath $UnitLog -ErrorAction SilentlyContinue)
                        $Count = if ($PrintedUnitLines.ContainsKey($UnitLog)) { $PrintedUnitLines[$UnitLog] } else { 0 }
                        if ($Lines.Count -gt $Count) {
                            $Lines[$Count..($Lines.Count - 1)] | Write-Output
                            $PrintedUnitLines[$UnitLog] = $Lines.Count
                        }
                    }
                    if ($Exited) { $ActiveShards.Remove($Shard.Slot) }
                }
                if ($UnitClock.Elapsed.TotalSeconds -ge 900) { throw 'Unit suite exceeded 900 seconds.' }
            } while ($NextModule -lt $PhaseModules.Count -or $ActiveShards.Count -gt 0)
        }
        $UnitExit = 0
        $ActualTests = 0
        foreach ($Shard in $Shards) {
            Write-Output "Unit process $($Shard.Process.Id): exit $($Shard.Process.ExitCode)."
            if ($Shard.Process.ExitCode -ne 0) { $UnitExit = 1 }
            $Summary = Select-String -LiteralPath $Shard.Stderr -Pattern '^Ran (\d+) tests? in ' | Select-Object -Last 1
            if ($null -eq $Summary) { $UnitExit = 1 } else { $ActualTests += [int]$Summary.Matches[0].Groups[1].Value }
        }
        Write-Output "Completed $ActualTests of $ExpectedTests discovered tests."
        if ($ActualTests -ne $ExpectedTests) { $UnitExit = 1 }
        [ordered]@{
            expected = $ExpectedTests; completed = $ActualTests; passed = ($UnitExit -eq 0)
            diagnostic = [bool]$IdleDiagnostic
        } |
            ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $OutputDirectory 'unit-report.json')
    } finally {
        foreach ($Shard in $Shards) {
            try {
                if (-not $Shard.Process.HasExited) {
                    $Shard.Process.Kill($true)
                    if (-not $Shard.Process.WaitForExit(30000)) { throw 'Unit process tree did not retire.' }
                }
                $Shard.Process.Dispose()
            } catch {
                $UnitExit = 1
                Write-Warning $_
            }
        }
        $TimingsPath = Join-Path $OutputDirectory 'timings.json'
        $Timings = Get-Content -Raw $TimingsPath | ConvertFrom-Json -AsHashtable
        $Timings['unit'] = [Math]::Round($UnitClock.Elapsed.TotalSeconds, 3)
        $Timings | ConvertTo-Json | Set-Content -Encoding utf8 $TimingsPath
    }
    exit $UnitExit
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
$AcceptanceStarted = [DateTime]::Now
$Password = ConvertTo-SecureString ('Rc!9' + [guid]::NewGuid().ToString('N')) -AsPlainText -Force

function Save-Progress([string] $Stage) {
    # Keep diagnostics in the checkout before any potentially blocking cleanup.
    # A step deadline can then retain them even if an OS operation never returns.
    $Processes = @(Get-Process -ErrorAction SilentlyContinue)
    $Details = @($Processes | Where-Object {
        $_.ProcessName -in @('python', 'pwsh', 'conhost', 'MsMpEng', 'Runner.Worker', 'Runner.Listener')
    } | Select-Object Id, ProcessName, CPU, WorkingSet64, HandleCount,
        @{ Name = 'ThreadCount'; Expression = { $_.Threads.Count } } -ErrorAction SilentlyContinue)
    $OperatingSystem = Get-CimInstance Win32_OperatingSystem
    $Counters = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $Memory = [ordered]@{
        physical_total_bytes = [long]$OperatingSystem.TotalVisibleMemorySize * 1024
        physical_free_bytes = [long]$OperatingSystem.FreePhysicalMemory * 1024
        committed_bytes = $Counters.CommittedBytes
        commit_limit_bytes = $Counters.CommitLimit
        pool_nonpaged_bytes = $Counters.PoolNonpagedBytes
        pool_paged_bytes = $Counters.PoolPagedBytes
    }
    $Logs = @()
    if ($null -ne $ChildOutput -and (Test-Path -LiteralPath $ChildOutput)) {
        foreach ($Log in @(Get-ChildItem -LiteralPath $ChildOutput -File -Filter 'unit-*.*')) {
            $Logs += [ordered]@{ name = $Log.Name; bytes = $Log.Length }
            Copy-Item -LiteralPath $Log.FullName -Destination $Reports -Force
        }
    }
    [ordered]@{
        utc = [DateTime]::UtcNow.ToString('o')
        stage = $Stage
        process_count = $Processes.Count
        processes = $Details
        memory = $Memory
        disks = @(Get-PSDrive -PSProvider FileSystem | Select-Object Name, Used, Free)
        tcp_states = @([Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpConnections() |
            Group-Object State | Select-Object Name, Count)
        unit_logs = $Logs
    } | ConvertTo-Json -Depth 5 -Compress |
        Add-Content -Encoding utf8 (Join-Path $Reports 'progress.jsonl')
    # Threat/block evidence is diagnostic only; absent logs never affect acceptance.
    try {
        $Events = @(Get-WinEvent -FilterHashtable @{
            LogName = 'Microsoft-Windows-Windows Defender/Operational'
            Id = @(1116, 1117, 1121, 1122)
            StartTime = $AcceptanceStarted
        } -MaxEvents 20 -ErrorAction SilentlyContinue |
            Select-Object TimeCreated, Id, Message)
        [ordered]@{ checked_utc = [DateTime]::UtcNow.ToString('o'); events = $Events } |
            ConvertTo-Json -Depth 4 |
            Set-Content -Encoding utf8 (Join-Path $Reports 'defender-events.json')
    } catch {
        Write-Verbose "Defender event diagnostics unavailable: $($_.Exception.Message)"
    }
    Write-Output "Windows acceptance: $Stage; $($Processes.Count) processes at $([DateTime]::UtcNow.ToString('o'))."
}

function Save-RunnerProvenance {
    $Evidence = @(foreach ($NativeProcess in @(Get-CimInstance Win32_Process |
            Where-Object { $_.Name -like 'provjobd*' -or $_.Name -eq 'provisioner.exe' })) {
        $Item = [ordered]@{
            pid = $NativeProcess.ProcessId
            parent_pid = $NativeProcess.ParentProcessId
            name = $NativeProcess.Name
            path = $NativeProcess.ExecutablePath
            created = $NativeProcess.CreationDate
        }
        try {
            $Owner = Invoke-CimMethod -InputObject $NativeProcess -MethodName GetOwner
            $Item['owner'] = "$($Owner.Domain)\$($Owner.User)"
            if ($NativeProcess.ExecutablePath) {
                $Item['sha256'] = (Get-FileHash -LiteralPath $NativeProcess.ExecutablePath -Algorithm SHA256).Hash
                $Signature = Get-AuthenticodeSignature -LiteralPath $NativeProcess.ExecutablePath
                $Item['signature_status'] = $Signature.Status.ToString()
                $Item['signer'] = if ($Signature.SignerCertificate) { $Signature.SignerCertificate.Subject } else { $null }
                $Item['signer_thumbprint'] = if ($Signature.SignerCertificate) { $Signature.SignerCertificate.Thumbprint } else { $null }
            }
        } catch {
            $Item['evidence_error'] = $_.Exception.Message
        }
        $Item
    })
    [ordered]@{ captured_utc = [DateTime]::UtcNow.ToString('o'); processes = $Evidence } |
        ConvertTo-Json -Depth 5 |
        Set-Content -Encoding utf8 (Join-Path $Reports 'runner-provenance.json')
}

function Save-DefenderState([string] $Name) {
    $Status = Get-MpComputerStatus | Select-Object AMRunningMode, AMServiceEnabled,
        AMProductVersion, AMEngineVersion, AntivirusEnabled, AntivirusSignatureVersion,
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

if ($InspectHost) {
    Save-RunnerProvenance
    $null = Save-DefenderState 'inherited'
    $Password.Dispose()
    exit 0
}

try {
    Save-Progress 'provisioning'
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
    Save-RunnerProvenance
    $null = Save-DefenderState 'inherited'
    Write-Output "Updating Defender security intelligence at $([DateTime]::UtcNow.ToString('o'))."
    Start-Service WinDefend
    Update-MpSignature
    $null = Save-DefenderState 'after-signature-update'
    Save-Progress 'updated Defender security intelligence'
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
                ' -OutputDirectory "' + $ChildOutput + '"' +
                $(if ($IdleDiagnostic) { ' -IdleDiagnostic' } else { '' })
            Credential = $Credential
            LoadUserProfile = $true
            Environment = $ChildEnvironment
            WorkingDirectory = $Source
            RedirectStandardOutput = Join-Path $Reports "$Phase.stdout.log"
            RedirectStandardError = Join-Path $Reports "$Phase.stderr.log"
            PassThru = $true
        }
        Write-Output "Starting tests at $([DateTime]::UtcNow.ToString('o'))."
        Save-Progress 'starting standard-user process'
        $Process = Start-Process @Launch
        $Retired = $false
        Save-Progress "started standard-user process $($Process.Id)"
        # Short waits keep cancellation responsive and expose each Python stage.
        $Waiting = [Diagnostics.Stopwatch]::StartNew()
        $PrintedLines = @{}
        $BudgetMinutes = 20
        $SnapshotSeconds = 5
        $NextSnapshot = $SnapshotSeconds
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
            if ($Waiting.Elapsed.TotalSeconds -ge $NextSnapshot) {
                Save-Progress "running standard-user process $($Process.Id)"
                $NextSnapshot = $Waiting.Elapsed.TotalSeconds + $SnapshotSeconds
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
    try { Save-Progress 'entering cleanup' } catch { Write-Warning $_ }
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
