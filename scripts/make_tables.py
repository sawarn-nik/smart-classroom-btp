"""Turn results/metrics/*.json into LaTeX tables (results/tables/*.tex).

Tables follow the booktabs style already loaded by the paper template.
Missing result files are skipped (e.g. engagement before DAiSEE arrives).
"""
import json

from src.common.paths import METRICS, TABLES


def load(name):
    p = METRICS / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def write(name, body):
    (TABLES / name).write_text(body)
    print("wrote", name)


def attendance():
    d = load("attendance_lfw.json")
    if not d:
        return
    v, idn = d["verification"], d["identification"]
    rows = []
    for m, lab in (("facenet", "FaceNet (InceptionResNetV1, triplet)"), ("arcface", "ArcFace (ResNet-50)")):
        rows.append(f"{lab} & {pct(v[m]['acc'])} $\\pm$ {pct(v[m]['acc_std'])} & {pct(v[m]['tar@far=1e-3'])} & "
                    f"{pct(idn['k=1'][m]['rank1'])} & {pct(idn['k=3'][m]['rank1'])} & "
                    f"{pct(idn['k=3'][m]['tpir@fpir=0.01'])} \\\\")
    g = idn["k=3"]["arcface"]
    write("attendance_lfw.tex", rf"""\begin{{table}}[H]
\caption{{Face recognition on LFW. Verification: 6000 standard pairs, 10-fold. Identification: {g['n_gallery']} enrolled identities, $k$ enrolment images each, {g['n_impostors']} unenrolled impostor probes.}}
\centering
\label{{tab:att_lfw}}
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{lccccc}}
\toprule
\multirow{{2}}{{*}}{{\bf Embedding}} & \multicolumn{{2}}{{c}}{{\bf Verification}} & \multicolumn{{3}}{{c}}{{\bf Identification}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-6}}
 & Acc. (\%) & TAR@FAR=$10^{{-3}}$ & Rank-1, $k$=1 & Rank-1, $k$=3 & TPIR@FPIR=1\% \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table}}
""")

    rob = d["robustness"]
    nice = {"clean": "Clean", "occlusion": "Lower-face occlusion", "occlusion_eyes": "Eye occlusion",
            "blur": "Blur ($k$=9)", "blur_strong": "Blur ($k$=15)", "low_light": "Low light",
            "low_res": "Face $\\approx$28\\,px", "low_res_20": "Face $\\approx$20\\,px", "low_res_14": "Face $\\approx$14\\,px"}
    rows = [f"{nice[c]} & {pct(r['facenet']['rank1'])} & {pct(r['facenet']['tpir@fpir=0.01'])} & "
            f"{pct(r['arcface']['rank1'])} & {pct(r['arcface']['tpir@fpir=0.01'])} \\\\" for c, r in rob.items()]
    write("attendance_robustness.tex", rf"""\begin{{table}}[H]
\caption{{Identification under synthetic classroom corruptions applied to probe images (gallery: 3 clean images per identity; up to 3 probes per identity).}}
\centering
\label{{tab:att_rob}}
\begin{{tabular}}{{lcccc}}
\toprule
\multirow{{2}}{{*}}{{\bf Condition}} & \multicolumn{{2}}{{c}}{{\bf FaceNet}} & \multicolumn{{2}}{{c}}{{\bf ArcFace}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}}
 & Rank-1 (\%) & TPIR@1\% & Rank-1 (\%) & TPIR@1\% \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def classroom_sim():
    d = load("attendance_classroom_sim_det1280.json")
    if not d:
        return
    a, c = d["aggregate"], d["config"]
    rows = []
    for m, lab in (("facenet", "FaceNet"), ("arcface", "ArcFace")):
        by = " / ".join(pct(x, 1) for x in a[f"{m}_recall_by_row_front_to_back"])
        rows.append(f"{lab} & {pct(a[f'{m}_precision'])} & {pct(a[f'{m}_recall'])} & {pct(a[f'{m}_f1'])} & "
                    f"{pct(a[f'{m}_exact_roll'], 1)} & {by} \\\\")
    px = " / ".join(str(int(x)) for x in a["face_px_by_row_front_to_back"])
    write("attendance_classroom_sim.tex", rf"""\begin{{table}}[H]
\caption{{End-to-end attendance on {c['sessions']} synthetic classroom frames (1920$\times$1080, {c['class_size']} enrolled students, 70--95\% present, 3 unenrolled visitors). Exact roll = session with no attendance error. Face widths per row (front to back): {px}\,px.}}
\centering
\label{{tab:att_sim}}
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{lccccc}}
\toprule
\bf Embedding & \bf Precision (\%) & \bf Recall (\%) & \bf F1 (\%) & \bf Exact roll (\%) & \bf Recall by row (\%) \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table}}
""")


def activity():
    d = load("activity_scb.json")
    if not d:
        return
    run = next(iter(d))
    r = d[run]
    rows = [f"{c.replace('_', ' ').capitalize()} & {pct(x['P'], 1)} & {pct(x['R'], 1)} & {pct(x['AP50'], 1)} & {pct(x['AP50-95'], 1)} \\\\"
            for c, x in r["source_aware_per_class"].items()]
    write("activity_scb.tex", rf"""\begin{{table}}[H]
\caption{{Behaviour detection with fine-tuned YOLOv8n ({r['params_M']:.1f}\,M parameters) on the held-out SCB test split. Each class is scored on test images from the SCB subset that annotates it.}}
\centering
\label{{tab:act}}
\begin{{tabular}}{{lcccc}}
\toprule
\bf Class & \bf P (\%) & \bf R (\%) & \bf AP$_{{50}}$ (\%) & \bf AP$_{{50:95}}$ (\%) \\
\midrule
{chr(10).join(rows)}
\midrule
Mean (source-aware) & & & {pct(r['source_aware_mAP50'], 1)} & {pct(r['source_aware_mAP50-95'], 1)} \\
All test images (naive) & & & {pct(r['naive_all_images']['mAP50'], 1)} & {pct(r['naive_all_images']['mAP50-95'], 1)} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def latency():
    w = load("pipeline_latency_det640.json")
    s = load("pipeline_steady_state.json")
    if not w or not s:
        return
    L = w["latency_ms"]
    keys = s["runs"][0]["steady_breakdown_ms"].keys()
    steady = {k: sum(r["steady_breakdown_ms"][k] for r in s["runs"]) / len(s["runs"]) for k in keys}
    nice = {"detect": "Face detection (SCRFD)", "track": "Tracking", "attendance": "Attendance (ArcFace)",
            "engagement": "Engagement cues (MediaPipe)", "activity": "Behaviour detection (YOLOv8n)",
            "total": "\\bf Total"}
    rows = [f"{nice[k]} & {L[k]['median']:.1f} & {steady[k]:.1f} \\\\" for k in nice]
    write("pipeline_latency.tex", rf"""\begin{{table}}[H]
\caption{{Per-frame latency (ms, median) on an Apple M1 CPU (YOLO on the M1 GPU). Worst case: {w['frames']} unrelated SCB classroom photos (mean {w['faces_per_frame_mean']:.1f} faces/frame), every face identified each frame. Steady state: static camera, identities cached after confirmation (mean of {len(s['runs'])} clips with 17--40 students).}}
\centering
\label{{tab:latency}}
\begin{{tabular}}{{lcc}}
\toprule
\bf Stage & \bf Worst case & \bf Steady state \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def engagement():
    d = load("engagement_daisee.json")
    if not d:
        print("engagement: no DAiSEE results yet")
        return
    r3, r4 = d["results"]["3class"], d["results"]["4class"]
    nice = {"majority": "Majority class", "rule_based": "Rule-based (pose/eyes)", "logreg": "Logistic regression",
            "gbm": "Gradient boosting", "bigru": "BiGRU (temporal)"}
    rows = []
    for k, lab in nice.items():
        a = r3.get(k)
        b = r4.get(k)
        f = lambda x, key: pct(x[key], 1) if x else "--"
        rows.append(f"{lab} & {f(a, 'acc')} & {f(a, 'macro_f1')} & {f(b, 'acc')} & {f(b, 'macro_f1')} \\\\")
    write("engagement_daisee.tex", rf"""\begin{{table}}[H]
\caption{{Engagement recognition on the DAiSEE test split ({d['n']['test']} clips).}}
\centering
\label{{tab:eng}}
\begin{{tabular}}{{lcccc}}
\toprule
\multirow{{2}}{{*}}{{\bf Model}} & \multicolumn{{2}}{{c}}{{\bf 3-class}} & \multicolumn{{2}}{{c}}{{\bf 4-class}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}}
 & Acc. (\%) & Macro-F1 (\%) & Acc. (\%) & Macro-F1 (\%) \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
""")
    abl = r3["ablation_gbm"]
    rows = [f"{k.replace('au', 'AU').replace('pose', 'Pose').replace('gaze', 'Gaze').replace('+', ' + ')} & {pct(v['acc'], 1)} & {pct(v['macro_f1'], 1)} \\\\"
            for k, v in abl.items()]
    write("engagement_ablation.tex", rf"""\begin{{table}}[H]
\caption{{Feature-group ablation (gradient boosting, 3-class, DAiSEE test).}}
\centering
\label{{tab:eng_abl}}
\begin{{tabular}}{{lcc}}
\toprule
\bf Features & \bf Acc. (\%) & \bf Macro-F1 (\%) \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


if __name__ == "__main__":
    attendance(); classroom_sim(); activity(); latency(); engagement()
