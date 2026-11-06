"""Bounded feedback/rejection feasibility probe. No novel neural task solution.

Dependencies: Python stdlib, local arena.py/isolation.py and their transitive
NC-007 frozen Torch model, NC-005/006 execution weights, Linux Landlock and
libseccomp. Neither oracle formulas nor expected values enter proposer input.
"""
import hashlib
import json
from pathlib import Path
import arena

HERE=Path(__file__).resolve().parent

def run_goal(goal, *, fault_injection=False):
    seed=11
    heldout=arena.pairs(seed,goal)
    rejected=[]
    attempts=[]
    for index in range(2):
        # The only feedback to the frozen proposer is rejection of earlier
        # complete programs; no heldout label or oracle output is passed in.
        request={"seed":seed,"goal":goal}
        if rejected:
            request["rejected_actions"]=rejected
        proposal,cost=arena.isolated("propose_worker.py",request)
        status=proposal.get("status") if proposal else "PROCESS_FAILED"
        entry={"attempt":index+1,"proposal_status":status,
               "proposal_request_fields":sorted(request),
               "proposal_cost":cost,
               "model_generated":bool(proposal and proposal.get("model_generated")),
               "learned_feedback":False,
               "source_class":"supervised_frozen_neural_controller",
               "actions":proposal.get("actions",[]) if proposal else [],
               "correct":0,"tested":len(heldout),
               "counterexamples":[]}
        if status!="PROPOSED":
            entry["result"]="CANNOT_PROPOSE"
            attempts.append(entry)
            break
        entry["action_sha256"]=hashlib.sha256(
            json.dumps(entry["actions"],separators=(",",":")).encode()).hexdigest()
        result,exec_cost=arena.isolated("execute_worker.py",{
            "seed":seed,"actions":entry["actions"],"pairs":heldout})
        entry["execution_cost"]=exec_cost
        outputs=list(result.get("outputs",[])) if result else []
        if fault_injection and index==0 and outputs:
            outputs[0]^=1  # synthetic measurement fault, explicitly labeled
            entry["negative_control"]="INJECTED_FIRST_OBSERVATION_BIT_FLIP"
        for i,(x,y) in enumerate(heldout):
            actual=outputs[i] if i<len(outputs) else None
            target=arena.truth(goal,x,y)
            if actual!=target:
                entry["counterexamples"].append(
                    {"pair_index":i,"x":x,"y":y,"expected":target,"observed":actual})
        entry["correct"]=len(heldout)-len(entry["counterexamples"])
        entry["result"]="MATCHES_FINITE_SAMPLES" if not entry["counterexamples"] else "COUNTEREXAMPLE"
        attempts.append(entry)
        if not entry["counterexamples"]:
            break
        rejected.append(entry["actions"])
    return {"goal":goal,"seed":seed,"fault_injection":fault_injection,
            "heldout_pairs_sha256":hashlib.sha256(json.dumps(heldout).encode()).hexdigest(),
            "attempts":attempts,"revisions":max(0,len(attempts)-1),
            "novel_discoveries":0,"learned_feedback":False}

def main():
    receipt=json.loads((HERE/"../NC-007/composition_receipt.json").read_text())
    for row in receipt["training"]:
        path=arena.ROOT/row["checkpoint_path"]
        if arena.hash_file(path)!=row["checkpoint_sha256"]:
            raise RuntimeError("frozen controller provenance mismatch")
    champion=arena.ROOT.parent/"rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt"
    before=arena.hash_file(champion)
    inventory=json.loads((HERE/"../NC-000/inventory.json").read_text())
    if before != inventory["champion"]["selected_bundle_sha256"]:
        raise RuntimeError("champion provenance mismatch")
    rows=[run_goal("DOUBLE"),run_goal("UNSEEN_XOR_DOUBLE_PLUS_Y"),
          run_goal("DOUBLE",fault_injection=True)]
    if arena.hash_file(champion)!=before:
        raise RuntimeError("champion mutated")
    assert rows[1]["attempts"][0]["proposal_status"]=="UNSUPPORTED_GOAL"
    assert rows[2]["attempts"][0]["negative_control"]=="INJECTED_FIRST_OBSERVATION_BIT_FLIP"
    assert rows[2]["revisions"]==1
    output={"schema":"logos.nc028.feedback-probe/1",
            "status":"NO_DISCOVERY",
            "verification":"IMPLEMENTER_ONLY_NOT_INDEPENDENT",
            "control_type":"synthetic observation fault injection, not natural neural correction",
            "conclusion":"Reranked frozen model after rejection; no learned feedback or structurally novel goal",
            "champion_sha256_unchanged":before,
            "source_sha256":{n:arena.hash_file(HERE/n)
                             for n in ("feedback_trial.py","propose_worker.py","arena.py","execute_worker.py","isolation.py")},
            "trials":rows}
    (HERE/"feedback_probe.json").write_text(json.dumps(output,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":output["status"],"trials":[
        {"goal":r["goal"],"fault":r["fault_injection"],
         "statuses":[a["result"] for a in r["attempts"]],
         "revisions":r["revisions"]} for r in rows]},sort_keys=True))
if __name__=="__main__":main()
