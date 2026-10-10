# Reference audit of mybibfile.bib (checked 2026-10-11 via Crossref / doi.org / arXiv)

## Fabricated: these papers do not exist (DOI 404, title not found). Remove them.
lit_facerecog_transformer_2025, lit_engagement_multimodal_2025, lit_drowsiness_yolo_2025,
lit_privacy_federated_2025, lit_emotion_gnn_2025, lit_pose_attention_2025,
lit_activity_recognition_2025, lit_transformer_attendance_2025, lit_edge_inference_2025,
lit_llm_analytics_2025, engagenet (the Abedi & Khan ICIP 2021 version)

Paper text that depends on them: literature survey lines ~132-150, 3 of the 4 rows in the
comparison table (~448-457), and the "only system in the literature" claim.

## Real papers with wrong metadata AND wrong claims in the .tex
| key | real authors / venue | what the paper actually says |
|---|---|---|
| attendance_android | Sunaryono, Siswantoro, Anggoro; vol 33(3) 304-312, 2021 | phone captures face + QR code; server-side LDA; 97.29% |
| engagement_springer | Das & Dev; Neural Comput. Appl. 37(23):19063-19085, 2025 | WACV + DAiSEE, behavioural features + AUs; XGBoost 82.9%, EfficientNet 47.2% |
| ensemble_ieee | Venugopal A, Krishna, Varma U; ICCCNT 2021, 10.1109/ICCCNT51525.2021.9580098 | ensemble of VGG-Face + FaceNet + OpenFace + DeepFace |
| disengagement_ieee | Pyari, Swapna, Gupta, Uma Lakshmi, Santosh, Kiran Bala; ETAACT 2026, 10.1109/ETAACT69135.2026.11542033 | STAG-Net (spatio-temporal attention graph), 96.8% acc, F1 95.1 |
| robotic_iot | Vinod Kumar, Ramesh, Pareek, Deepak, Ananda Babu; ICAISC | ResNet-18/50 on FER2013 + student faces (RFID only as a comparison) |

## Fine / minor fixes
resnet (add DOI 10.1109/CVPR.2016.90), mtcnn (should be @article, IEEE SPL), openface (ok),
arcface (TPAMI version is missing authors Jing Yang and Irene Kotsia), dav2/DAiSEE (title is "...Recognition in the Wild", arXiv 1609.01885),
yolov8 (cite as software: Jocher, Chaurasia, Qiu, 2023. "60 fps on CPU" is not supported; measure it ourselves).

## Verified real papers to use instead
- ArcFace CVPR 2019, 10.1109/CVPR.2019.00482; FaceNet CVPR 2015, 10.1109/CVPR.2015.7298682
- SCRFD (Guo et al.), arXiv 2105.04714
- EngageNet: Singh et al., ICMI 2023, 10.1145/3577190.3614164 (arXiv 2302.00431)
- Abedi & Khan, ResNet+TCN, CRV 2021, 10.1109/CRV52889.2021.00028 (arXiv 2104.10122)
- Liao et al., DFSTN, Applied Intelligence 2021, 10.1007/s10489-020-02139-8
- Selim et al., IEEE Access 2022, 10.1109/ACCESS.2022.3206779
- Gunawan et al., YOLOv9 attention monitoring, ICSIMA 2024, 10.1109/ICSIMA62563.2024.10675573
- Preuveneers & Joosen, IEEE EDGE 2019, 10.1109/EDGE.2019.00017
- SCB-Dataset arXiv 2304.02488; SCB-Dataset3 arXiv 2310.02522
- OpenFace 2.0, 10.1109/FG.2018.00019; MediaPipe Face Mesh arXiv 1907.06724; Attention Mesh arXiv 2006.10962

## Real DAiSEE 4-class accuracies (comparison baselines; re-check against each PDF before quoting)
Gupta et al. 2016 (LRCN 57.9, C3D 56.1, InceptionNet 46.4) · Liao 2021 DFSTN 58.84 ·
Abedi & Khan 2021 ResNet+TCN 63.9 · Selim 2022 EfficientNetB7+LSTM 67.48 → realistic range is ~47-67%.
