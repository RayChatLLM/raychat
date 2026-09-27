#!/bin/zsh
emulate -R sh
setopt KSH_ARRAYS
unsetopt POSIX_STRINGS
zmodload zsh/system || exit 70
# The launcher supplies anonymous pipe ends and immutable bootstrap metadata.
# No Python interpreter remains resident in this terminal guardian.
export LC_ALL=C
umask 077
: "${RAYCHAT_GUARDIAN_PYTHON:?}"
: "${RAYCHAT_GUARDIAN_ENTRY:?}"
: "${RAYCHAT_GUARDIAN_DIR:?}"
export RAYCHAT_GUARDIAN_CORE_LOG="$RAYCHAT_GUARDIAN_DIR/core.log"
RAYCHAT_GUARDIAN_SAVED_STTY=$(/bin/stty -g <&0) || exit 70
export RAYCHAT_GUARDIAN_SAVED_STTY
printf '%s\n' "$RAYCHAT_GUARDIAN_SAVED_STTY" >"$RAYCHAT_GUARDIAN_DIR/saved-stty"
printf '%s\n' "$ZSH_VERSION" >"$RAYCHAT_GUARDIAN_DIR/zsh-version"
core_pid= terminal_active=0
pending_bytes=0 queue_head=0 queue_tail=0
queue=()
dle=$'\020'

cleanup() {
    local exit_status=$?
    # A signal may arrive after the only background launch but before assignment.
    local owned_pid=${core_pid:-$!}
    trap - EXIT HUP TERM INT PIPE
    if [[ -n $owned_pid && $owned_pid -gt 1 && -z ${RAYCHAT_GUARDIAN_CORE_EXIT_STATUS+x} ]]; then
        kill -TERM "$owned_pid" 2>/dev/null
        # Cleanup remains bounded even for a stopped child.
        kill -KILL "$owned_pid" 2>/dev/null
        wait "$owned_pid" 2>/dev/null
    fi
    if [[ $terminal_active == 1 ]]; then
        printf '\033[?1006l\033[?1002l\033[?1000l\033[?2004l\033[?7h\033[?25h\033[?1049l'
        /bin/stty "$RAYCHAT_GUARDIAN_SAVED_STTY" <&0
    fi
    exit "$exit_status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 143' TERM
trap 'exit 130' INT
# A broken core pipe is a promotion/failure condition, not sudden guardian death.
trap '' PIPE
trap ':' CHLD WINCH

encode_input() {
    encoded=${1//"$dle"/"${dle}D"}
    encoded=${encoded//$'\n'/"${dle}N"}
    encoded=${encoded//$'\0'/"${dle}Z"}
}

save_pending() {
    local n
    : >"$RAYCHAT_GUARDIAN_DIR/pending-input.encoded"
    for ((n=queue_head; n<queue_tail; n++)); do
        printf '%s' "${queue[n]}" >>"$RAYCHAT_GUARDIAN_DIR/pending-input.encoded"
    done
}

promote() {
    local reason=$1
    printf '%s\n' "$reason" >"$RAYCHAT_GUARDIAN_DIR/promotion-reason"
    save_pending
    if [[ ! -e "$RAYCHAT_GUARDIAN_DIR/promotion-input.encoded" ]]; then
        : >"$RAYCHAT_GUARDIAN_DIR/promotion-input.encoded"
    fi
    if [[ -z ${RAYCHAT_GUARDIAN_LAUNCH_SHA256:-} ]]; then
        cat "$RAYCHAT_GUARDIAN_CORE_LOG" >&2
        exit 70
    fi
    # Same PID remains the core's parent; control output is still unread.
    # Promotion must own terminal restoration, these pipe FDs, and child reaping.
    # Caught shell traps reset at exec. Ignore termination only across the
    # exec/import boundary; guardian_entry installs an emergency owner first.
    trap '' HUP TERM INT
    exec "$RAYCHAT_GUARDIAN_PYTHON" -I -B -S "$RAYCHAT_GUARDIAN_ENTRY" adopt
    trap 'exit 129' HUP
    trap 'exit 143' TERM
    trap 'exit 130' INT
    exit 70
}

append_input() {
    [[ -z $1 ]] && return
    queue[queue_tail]=$1
    ((queue_tail+=1))
    ((pending_bytes+=${#1}))
    # Preserve the last bounded read as well, then promote before taking more.
    if ((pending_bytes>65536)); then promote input_overflow; fi
}

flush_input() {
    local body header packet written
    [[ -n ${RAYCHAT_GUARDIAN_LAUNCH_SHA256:-} ]] || return
    while ((queue_head<queue_tail)); do
        body=${queue[queue_head]}
        printf -v header 'I%08d;' "${#body}"
        packet=$header$body
        # Direct syscall builtin reports bytes committed, without stdio retries
        # or buffered data leaking onto the terminal after descriptor restoration.
        written=0
        syswrite -c written -o3 "$packet" 2>/dev/null
        if ((written==0)); then return; fi
        if ((written!=${#packet})); then
            printf '%s\n' "$written/${#packet}" >"$RAYCHAT_GUARDIAN_DIR/partial-write-error"
            exit 74
        fi
        ((pending_bytes-=${#body}))
        ((queue_head+=1))
    done
    queue_head=0 queue_tail=0
    queue=()
}

launch_args=("$@")
/bin/stty raw -echo min 1 time 0 <&0 || exit 70
terminal_active=1
printf '\033[?1049h\033[?25l\033[?2004h\033[?1000h\033[?1002h\033[?1006h\033[?7l\033[H'
"$RAYCHAT_GUARDIAN_PYTHON" -I -B -S "$RAYCHAT_GUARDIAN_ENTRY" prepare "$@" \
    <&5 4>&6 3>&- 5<&- 6>&- 2>>"$RAYCHAT_GUARDIAN_CORE_LOG" &
core_pid=$!
exec 5<&- 6>&-
export RAYCHAT_GUARDIAN_CORE_PID=$core_pid
printf '%s\n' "$core_pid" >"$RAYCHAT_GUARDIAN_DIR/core-pid"
while :; do
    if ! kill -0 "$core_pid" 2>/dev/null; then
        wait "$core_pid"
        export RAYCHAT_GUARDIAN_CORE_EXIT_STATUS=$?
        printf '%s\n' "$RAYCHAT_GUARDIAN_CORE_EXIT_STATUS" >"$RAYCHAT_GUARDIAN_DIR/core-exit-status"
        if [[ $RAYCHAT_GUARDIAN_CORE_EXIT_STATUS == 0 && -e "$RAYCHAT_GUARDIAN_DIR/finished" ]]; then exit 0; fi
        promote core_exit
    fi
    if [[ -z ${RAYCHAT_GUARDIAN_LAUNCH_SHA256:-} && -e "$RAYCHAT_GUARDIAN_DIR/ready" ]]; then
        IFS= read -r RAYCHAT_GUARDIAN_LAUNCH_SHA256 <"$RAYCHAT_GUARDIAN_DIR/launch-sha256" || exit 70
        IFS= read -r -d $'\0' prepared_root <"$RAYCHAT_GUARDIAN_DIR/bootstrap-root" || exit 70
        export RAYCHAT_GUARDIAN_LAUNCH_SHA256
        export RAYCHAT_GUARDIAN_ENTRY="$prepared_root/raychat_bootstrap/guardian_entry.py"
        : >"$RAYCHAT_GUARDIAN_DIR/bootstrap-ack"
    fi
    [[ -e "$RAYCHAT_GUARDIAN_DIR/promote" ]] && promote core_request
    flush_input
    chunk=
    sysread -i0 -s250 -t0.02 chunk
    read_status=$?
    if ((read_status==5)); then promote tty_eof; fi
    [[ -z $chunk ]] && continue
    if [[ $chunk == *$'\022'* ]]; then
        prefix=${chunk%%$'\022'*}
        suffix=${chunk#*$'\022'}
        encode_input "$prefix"
        append_input "$encoded"
        encode_input "$suffix"
        encoded=$'\022'"$encoded"
        printf '%s' "$encoded" >"$RAYCHAT_GUARDIAN_DIR/promotion-input.encoded"
        promote ctrl_r
    fi
    encode_input "$chunk"
    append_input "$encoded"
done
