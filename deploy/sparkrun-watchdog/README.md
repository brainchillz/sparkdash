# sparkrun-watchdog

> **Superseded (2026-10-13).** SparkDash now has an in-process watchdog
> (`sparkdash/watchdog.py`, README "Watchdog") that guards sparkrun recipes
> *and* engine stacks, with the same checks and the same cooldown, and shows
> its verdict in Admin → Settings. Where this timer is installed, disable it so
> the two do not both try to relaunch a job:
>
>     sudo systemctl disable --now sparkrun-watchdog.timer
>
> The files stay here for hosts that run sparkrun without SparkDash.

A systemd timer on the head node that notices a dead multi-node vLLM job and
relaunches it. Built after 2026-09-29, when bambam hard-reset, its container
vanished, and the remaining three ranks kept answering `/health` with 200 for
hours while no request could be served.

Every 2 minutes it checks, in order: every host in the job still has its
container up; `/health` answers once the job is past the startup grace; and a
tiny real chat completion comes back. Two consecutive failing runs make it
alert (through SparkDash's configured webhook), and if every host answers SSH
and no restart happened in the last 40 minutes it stops the job and relaunches
the same recipe with the same `-o` overrides.

Install (head node, as the sparkrun user):

    sudo install -m 0755 sparkrun-watchdog /usr/local/bin/
    sudo install -m 0644 sparkrun-watchdog.conf /etc/
    sudo install -m 0644 sparkrun-watchdog.service sparkrun-watchdog.timer /etc/systemd/system/
    sudo systemctl daemon-reload && sudo systemctl enable --now sparkrun-watchdog.timer

Dry run / logs:

    sparkrun-watchdog --dry-run
    journalctl -u sparkrun-watchdog -n 50
