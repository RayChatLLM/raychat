# CI-only permission fixture for the extracted application, never user setup.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $Root,
    [Parameter(Mandatory)][string] $ExpectedSid,
    [switch] $Writable
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $IsWindows -or $env:GITHUB_ACTIONS -ne 'true' -or
    $env:RUNNER_ENVIRONMENT -ne 'github-hosted') {
    throw 'This permission fixture requires an ephemeral GitHub-hosted Windows VM.'
}
$Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
if ($Identity.User.Value -ne $ExpectedSid -or
    'S-1-5-32-544' -in @($Identity.Groups.Value)) {
    throw 'The release fixture requires the expected ordinary-user identity.'
}
if (-not (Test-Path -LiteralPath (Join-Path $Root 'raychat') -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $Root '_raychat/raychat.py') -PathType Leaf)) {
    throw 'The target must be the extracted user release.'
}
$Acl = [Security.AccessControl.DirectorySecurity]::new()
$Acl.SetAccessRuleProtection($true, $false)
foreach ($Sid in @('S-1-5-18', 'S-1-5-32-544', $ExpectedSid)) {
    $Rights = if ($Sid -eq $ExpectedSid -and -not $Writable) {
        'ReadAndExecute'
    } else { 'FullControl' }
    $Acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new(
        [Security.Principal.SecurityIdentifier]::new($Sid), $Rights,
        'ContainerInherit,ObjectInherit', 'None', 'Allow'))
}
Set-Acl -LiteralPath $Root -AclObject $Acl
