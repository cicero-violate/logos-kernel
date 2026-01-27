#!/usr/bin/env bash
shopt -s expand_aliases
source ~/.bashrc

echo "you may 1) induce change edit with apply_patch, using valid apply_patch files in codeblock 2) comment 3) do nothing 4) update Goal-short-term.md"
# echo "Given the current Reality and its Truth representation, which invariant in your Structure most constrains the Kernel’s executable space, and within that constraint set, what alternative Actions remain reachable through Free Will that Judgment has not yet eliminated?"
# echo "how to use mod test without cargo"

pwd
ls -la /home/cicero-arch-omen/ai_sandbox/project_template/docs/Goal-short-term.md
bat -n docs/Goal-short-term.md
bat -n build.sh
wc_all *.rs
bat_all *.rs




