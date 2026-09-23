# Βιβλιογραφικός έλεγχος — μέθοδος I3 (εικόνα intensity ως deskew + αρχική θέση ICP)

Ημερομηνία: 2026-09-19. Όλες οι αναφορές ελέγχθηκαν με ανάκτηση της σελίδας που δίνεται στο URL
(arXiv / ar5iv / GitHub / PMC / Semantic Scholar API). Όπου η ανάκτηση απέτυχε ή έδωσε μόνο
μεταδεδομένα, σημειώνεται ρητά «⚠️ μερική επαλήθευση». Καμία αναφορά δεν προέρχεται από μνήμη.

Η δική μας μέθοδος (για θέση): LiDAR-only, χειροκίνητο Hesai QT64 (Oxford Spires). Πανόραμα
intensity 64×1024 από **ακατέργαστα** (undeskewed) σημεία· κάθε pixel κρατά 3D σημείο + χρόνο.
SIFT μεταξύ διαδοχικών scans → 3D ζεύγη με χρόνο → RANSAC (Kabsch) → προσαρμογή μοντέλου
σταθερής ταχύτητας (στροφή με γωνιακή επιτάχυνση) με robust LS στον χρόνο κάθε σημείου.
Η κίνηση αυτή αντικαθιστά την πρόβλεψη σταθερής ταχύτητας του KISS-ICP και για deskew και ως
αρχική θέση ICP· σ σταθερό. Φίλτρο για «συνταξιδεύοντα» μοτίβα στο κοντινό δάπεδο.
Αποτέλεσμα: ATE −59…−96 % έναντι KISS-SLAM σε 3 ακολουθίες (keble-college-02: 2.13 → 0.09 m).

---

## Ε1. Εκτίμηση κίνησης / deskew από εικόνες intensity LiDAR

| Εργασία | Venue / έτος | Αισθητήρες (IMU?) | Χρόνος ανά σημείο για κίνηση **εντός** scan? | Τροφοδοτεί deskew? | Σημειώσεις |
|---|---|---|---|---|---|
| Dong, Barfoot — *Lighting-Invariant VO using Lidar Intensity Imagery and Pose Interpolation* — [DOI 10.1007/978-3-642-40686-7_22](https://api.semanticscholar.org/graph/v1/paper/6d03ea6de7b0dfbfe3ead57ab976b99927d09e17?fields=title,authors,year,venue,abstract,externalIds) | FSR 2012 (Springer Tracts 2013) | Σαρωτικό lidar· IMU: **δεν επαληθεύτηκε** | **Ναι** (κατά τίτλο: pose interpolation· το snippet αναζήτησης λέει ότι «data acquired during continuous vehicle motion suffer from geometric motion distortion») | Ναι — η ίδια η εκτίμηση VO είναι συνεχούς χρόνου | ⚠️ μερική επαλήθευση: Springer PDF απαιτεί cookie· μόνο μεταδεδομένα (Semantic Scholar). **Ο πλησιέστερος πρόγονος** της ιδέας μας (χαρακτηριστικά σε εικόνα intensity + παρεμβολή θέσης). Όχι spinning 3D LiDAR, όχι ICP. |
| Barfoot et al. — *Into Darkness: Visual Navigation Based on a Lidar-Intensity-Image Pipeline* — [PDF ASRL](https://asrl.utias.utoronto.ca/~jdg/sbib/barfoot_isrr13.pdf) | ISRR 2013 | Σαρωτικό lidar· αναφέρει διορθώσεις στάσης από «celestial» (⚠️ IMU ασαφές) | Ναι — «continuous-time estimation for pose», «motion-compensated RANSAC» | Ναι (VO συνεχούς χρόνου) | Ίδια ομάδα (UTIAS). SURF σε εικόνες intensity. Teach-and-repeat, όχι ICP. |
| Wang, Wang, Xie — *Intensity-SLAM* — [arXiv 2102.03798](https://ar5iv.labs.arxiv.org/html/2102.03798) | RA-L 6(2), 2021 | LiDAR (VLP-16, KITTI)· IMU δεν αναφέρεται | Όχι | Όχι (deskew δεν αναφέρεται) | Όχι εικόνα: 3D πλέγμα intensity + intensity residual + intensity-weighted features. |
| Park, Jang, Kim — *I-LOAM* — [IEEE 9144987](https://ieeexplore.ieee.org/document/9144987/) | UR 2020 | LiDAR (KITTI) | Όχι | Όχι | Intensity στο cost της LOAM. ⚠️ μόνο abstract μέσω αναζήτησης. |
| Li et al. — *InTEn-LOAM* — [arXiv 2209.05708](https://arxiv.org/abs/2209.05708) | arXiv 2022 | LiDAR· IMU δεν αναφέρεται | Όχι — το «temporal» αφορά αφαίρεση κινούμενων αντικειμένων | Όχι | Κυλινδρικές εικόνες για εξαγωγή features + intensity-based registration. |
| Guadagnino et al. — *Fast Sparse LiDAR Odometry Using Self-Supervised Feature Selection on Intensity Images* — [DOI 10.1109/LRA.2022.3184454](https://api.semanticscholar.org/graph/v1/paper/search?query=Fast+Sparse+LiDAR+Odometry+Using+Self-Supervised+Feature+Selection+on+Intensity+Images&fields=title,abstract,venue,year,authors,externalIds&limit=3) | RA-L 7(3), 2022 | LiDAR-only (abstract: κανένα IMU) | Όχι (abstract: relative pose μεταξύ scans) | Όχι (δεν αναφέρεται) | Learned keypoints σε κυλινδρική εικόνα intensity → σχετική θέση. ⚠️ μόνο abstract (PDF 403/404). |
| Di Giammarino et al. — *MD-SLAM* — [arXiv 2203.13237](https://ar5iv.labs.arxiv.org/html/2203.13237) | IROS 2022 | LiDAR-only (και RGB-D)· χωρίς IMU | **Όχι** — «treats frames as instantaneous snapshots» | Όχι | Direct photometric σε εικόνες intensity+range+normals· Newer College. Καλύτερο σε σκάλες. |
| Du, Beltrame — *Real-Time SLAM with LiDAR intensity* — [arXiv 2301.09257](https://ar5iv.labs.arxiv.org/html/2301.09257) | ICRA 2023 | «pure LiDAR» (Ouster OS1-128 / OS0-64) | Όχι | Όχι (δεν αναφέρεται) | ORB σε εικόνα 1024×64, 3D σημείο μέσω index του pixel — **ίδια ιδέα lookup με τη δική μας**, αλλά χωρίς χρόνο/deskew. |
| Zhang et al. — *RI-LIO* — [GitHub README](https://github.com/RoboFeng/RI-LIO/blob/main/README.md) | RA-L 8(3), 2023, DOI 10.1109/LRA.2023.3243528 | **LiDAR + IMU** (iEKF, πάνω σε FAST-LIO2)· μόνο Ouster | Όχι από την εικόνα | Deskew από IMU (⚠️ συμπέρασμα από τη βάση FAST-LIO2, δεν το λέει το README) | Photometric residual εικόνας reflectivity + point-to-plane. |
| Pfreundschuh et al. — *COIN-LIO* — [arXiv 2310.01235](https://ar5iv.labs.arxiv.org/html/2310.01235) | ICRA 2024, σ. 1730–1737 | **LiDAR + IMU** (iEKF)· Ouster OS0-128 | Χρησιμοποιεί χρόνο ανά σημείο **μόνο** για να βρει το tracked σημείο στο παραμορφωμένο frame | **Όχι** — η εικόνα φτιάχνεται από IMU-undistorted σημεία· «relies entirely on IMU for ego-motion compensation» | Φίλτρο φωτεινότητας (brightness map σε μεγάλο παράθυρο) + αφαίρεση οριζόντιων γραμμών· επιλογή patches σε degenerate διευθύνσεις. |
| Zheng, Zhu — *ECTLO* — [arXiv 2206.08517](https://arxiv.org/abs/2206.08517) | IROS 2023 | LiDAR-only (Risley-prism) | Ναι (continuous-time μοντέλο, αλλά από **γεωμετρία**, όχι intensity) | Ναι | Εικόνα **range**, όχι intensity. |
| Ouster blog — *Lidar as a camera* — [ouster.com](https://ouster.com/insights/blog/the-camera-is-in-the-lidar) | blog (χωρίς ημερομηνία στη σελίδα) | — | Όχι | Όχι | SuperPoint σε εικόνες signal/depth· «zero temporal mismatch» **μεταξύ στρωμάτων** (η σάρωση παραμένει rolling). Όχι επιστημονική δημοσίευση. |

**Σύνοψη Ε1.** Όσες εργασίες εξάγουν χαρακτηριστικά από εικόνα intensity spinning 3D LiDAR
(Guadagnino 2022, MD-SLAM, Du & Beltrame 2023, RI-LIO, COIN-LIO) είτε αγνοούν την κίνηση εντός
scan είτε τη διορθώνουν με IMU. Καμία από αυτές δεν εκτιμά την **εντός-scan** κίνηση από τους
χρόνους των αντιστοιχισμένων pixels και δεν τη δίνει ως deskew σε ICP. Η μόνη γραμμή που κάνει
«χαρακτηριστικά intensity + παρεμβολή θέσης στον χρόνο» είναι η UTIAS (Dong & Barfoot 2012,
Barfoot et al. 2013) σε σαρωτικό (όχι spinning multi-beam) lidar, χωρίς ICP — και εκείνη
επαληθεύτηκε μόνο μερικώς.

---

## Ε2. Συνεχής χρόνος / deskew σε LiDAR-only οδομετρία

| Εργασία | Venue / έτος | IMU? | Μοντέλο κίνησης εντός scan | Χρόνος ανά σημείο? | Σημειώσεις |
|---|---|---|---|---|---|
| Vizzo et al. — *KISS-ICP* — [arXiv 2209.15397](https://ar5iv.labs.arxiv.org/html/2209.15397) | RA-L 2023 | Όχι | **Σταθερή ταχύτητα από τις δύο τελευταίες θέσεις** T_{t−1}, T_{t−2}: T_pred = [R_{t−2}ᵀR_{t−1}, R_{t−2}ᵀ(t_{t−1}−t_{t−2})]· deskew p*_i = Exp(s_i ω_t) p_i + s_i v_t· ίδια πρόβλεψη ως αρχική θέση ICP· τ_t = 3σ_t | Ναι (s_i ∈ [0, Δt]) | Δικαιολόγηση: «acceleration … within such short time intervals [10–20 Hz] are fairly small». KITTI-raw ablation (Table V): χωρίς deskew 0.91 %, σταθ. ταχύτητα 0.49 %, IMU 0.51 %. Newer College (handheld): 0.51 % / 0.96 % (short/long). |
| Dellenbach et al. — *CT-ICP* — [arXiv 2109.12979](https://ar5iv.labs.arxiv.org/html/2109.12979) | ICRA 2022, σ. 5580–5586 | Όχι | Δύο θέσεις ανά scan (αρχή/τέλος), slerp + γραμμική παρεμβολή· «elastic» deskew **μέσα** στην εγγραφή· ασυνέχεια μεταξύ scans | Ναι | Περιορισμοί C_loc, C_vel προς προηγούμενο scan. Newer College handheld. |
| Zheng, Zhu — *Traj-LO* — [arXiv 2309.13842](https://ar5iv.labs.arxiv.org/html/2309.13842) | RA-L 9(2), 2024 | Όχι | Piecewise-linear με K τμήματα και έλεγχο-θέσεις, κινηματική ομαλότητα | Ναι | «linear interpolation at a low-frequency scan rate becomes inaccurate when facing aggressive motion». Hilti 2021 handheld, NTU VIRAL, Point-LIO. |
| Burnett, Schoellig, Barfoot — *CT Radar/Lidar-Inertial Odometry using a GP Motion Prior* — [arXiv 2402.06174](https://ar5iv.labs.arxiv.org/html/2402.06174) | T-RO 2024 | STEAM-LO χωρίς IMU / STEAM-LIO με IMU | Gaussian-process prior (WNOA) | Ναι | Baseline σταθερής ταχύτητας: KITTI-raw 0.66 % vs 0.53 %· **Newer College 01-Short ATE 0.856 m (CV) vs 0.340 m (CT)**· «Similar to KISS-ICP, our constant-velocity baseline fails on the Dynamic Spinning sequence». |
| Deschênes, Vannini, Giguère, Pomerleau — *Stretch-ICP* — [arXiv 2605.17264](https://arxiv.org/html/2605.17264) | Sensors 26(8):2567, 2026 ✅ υπάρχει | **Ναι** (IMU preintegration, SAAVE για κορεσμένο γυροσκόπιο) | Συνεχής τροχιά που «τεντώνεται» κατά την εγγραφή | Ναι | «For all 32 runs of the TIGS dataset, KISS-ICP diverged … aggressive motions violate the constant-velocity assumption underlying KISS-ICP's deskewing model.» Εξαιρετικά βίαιη κίνηση (κύλισμα σε λόφο). |
| Han, Lee — *Deskewed LiDAR Odometry for Quadruped Robots …* — [PMC13259366](https://pmc.ncbi.nlm.nih.gov/articles/PMC13259366/), DOI 10.3390/s26113518 | Sensors 26(11):3518, 2026 ✅ υπάρχει | **Όχι** («LiDAR-only design philosophy») | Piecewise-constant velocity με clamping + ICP δύο σταδίων (οριζόντιο/κατακόρυφο) | Ναι | Unitree Go2 + VLP-16. «KISS-ICP assumes a constant-velocity motion model, which becomes unreliable under abrupt attitude changes.» **Δεν** έχει ablation no-deskew vs CV. |
| Potokar, Kaess — *A Comprehensive Evaluation of LiDAR Odometry Techniques* — [arXiv 2507.16000](https://arxiv.org/html/2507.16000) | IROS 2025 | Σύγκριση none / CV / IMU | — | — | «some form of dewarping was worthwhile, with IMU dewarping performing the best» αλλά «if computational expense is a constraint, one should consider using no dewarping since the gains for using constant velocity dewarping are more minimal». Newer College, Hilti 2022. |
| Malladi, Guadagnino, Lobefaro, Stachniss — *A Robust Approach for LiDAR-Inertial Odometry Without Sensor-Specific Modeling* — [arXiv 2509.06593](https://arxiv.org/html/2509.06593v1) | arXiv Σεπ. 2025 | LIO (σύγκριση με KISS-ICP) | — | — | Σε Oxford Spires: «KISS-ICP … uses a constant velocity motion assumption which can fail to sufficiently model the motion profile of a backpack-mounted sensor». Βλ. Ε5. |

**Τι ακριβώς υποθέτει το KISS-ICP (επαληθευμένο από το κείμενο):** η κίνηση εντός του τρέχοντος
scan = η σχετική κίνηση **μεταξύ των δύο προηγούμενων θέσεων**, εφαρμοσμένη γραμμικά στον χρόνο
s_i κάθε σημείου· η ίδια πρόβλεψη είναι και η αρχική θέση της ICP. Έχει αναφερθεί ότι
αποτυγχάνει σε: κύλισμα (Stretch-ICP, 32/32), Newer College «Dynamic Spinning» (Burnett 2024),
τετράποδο (Han & Lee 2026, ποιοτικά), Oxford Spires handheld (Malladi 2025, Keble ATE 9.48 m).

---

## Ε3. Έχει αναφερθεί ότι το deskew σταθερής ταχύτητας είναι **χειρότερο από καθόλου deskew**?

| Πηγή | Τι αναφέρει | Απαντά στο ερώτημα? |
|---|---|---|
| KISS-ICP (RA-L 2023), Table V | Σε **όχημα** (KITTI-raw) το CV deskew **βοηθά**: 0.91 → 0.49 %. | Όχι — αντίθετη πλατφόρμα. |
| Potokar & Kaess (IROS 2025) | Σε handheld datasets τα κέρδη του CV deskew είναι «more minimal»· προτείνουν «no dewarping» αν κοστίζει. **Δεν** αναφέρουν περίπτωση όπου CV < none. | Κοντά, αλλά όχι: λέει «μικρό κέρδος», όχι «ζημιά». |
| Burnett et al. (T-RO 2024) | CV baseline αποτυγχάνει σε Dynamic Spinning· 0.86 vs 0.34 m σε 01-Short. Σύγκριση CV vs **CT**, όχι CV vs none. | Όχι άμεσα. |
| Stretch-ICP (Sensors 2026) | KISS-ICP αποκλίνει 32/32 σε TIGS· δεν δοκιμάζεται KISS-ICP χωρίς deskew. | Όχι άμεσα. |
| Han & Lee (Sensors 2026) | Single-twist deskew «cannot adequately suppress» distortion τετραπόδου· χωρίς ablation none/CV. | Όχι άμεσα. |
| Malladi et al. (2025) | Η υπόθεση CV «can fail» για backpack· χωρίς ablation. | Όχι άμεσα. |

**Συμπέρασμα Ε3.** Δεν βρέθηκε (σε 4 στοχευμένες αναζητήσεις + ανάγνωση 6 εργασιών) δημοσιευμένη
μέτρηση ότι το deskew σταθερής ταχύτητας από το προηγούμενο scan είναι **χειρότερο από καθόλου
deskew** σε handheld/legged δεδομένα. Η βιβλιογραφία λέει «ανεπαρκές» ή «μικρό κέρδος». Η δική
μας διάγνωση (χειρότερο από τίποτα στο Oxford Spires) φαίνεται **μη αναφερθείσα** — με την
επιφύλαξη ότι η αναζήτηση δεν είναι εξαντλητική και ότι το εύρημα πρέπει να τεκμηριωθεί με
ablation (KISS-SLAM με deskew off) στο δικό μας log πριν διεκδικηθεί.

---

## Ε4. Αντιστοίχιση χαρακτηριστικών σε πανοράματα intensity — δάπεδο, εξάρτηση από απόσταση/γωνία, βαθμονόμηση

| Εργασία | Venue / έτος | Τι λέει | Σχέση με το δικό μας φίλτρο δαπέδου |
|---|---|---|---|
| Kashani, Olsen, Parrish, Wilson — *A Review of LIDAR Radiometric Processing* — [Europe PMC 26561813](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:26561813&format=json&resultType=core), DOI 10.3390/s151128099 | Sensors 15(11):28099–28128, 2015 | Επισκόπηση: το intensity εξαρτάται από απόσταση, γωνία πρόσπτωσης, ατμόσφαιρα, δέκτη· επίπεδα επεξεργασίας (raw → correction → normalization → calibration). | Θεωρητικό υπόβαθρο: το intensity του δαπέδου αλλάζει με R και α καθώς κινείται ο αισθητήρας. |
| Jutzi, Gross — *Normalization of LiDAR Intensity Data Based on Range and Surface Incidence Angle* — [ISPRS PDF](https://www.isprs.org/proceedings/xxxviii/3-w8/papers/213_laserscanning09.pdf) | Laserscanning 2009, IAPRS 38(3/W8):213–218 | Κανονικοποίηση ως προς R και cos α. ⚠️ PDF ανακτήθηκε αλλά δεν αναλύθηκε κειμενικά· μεταδεδομένα από αναζήτηση. | Κλασική διόρθωση· δεν λύνει το «συνταξιδεύον» μοτίβο (βλ. κάτω). |
| Viswanath, Jiang, Saripalli — *Reflectivity Is All You Need!* — [arXiv 2403.13188](https://arxiv.org/html/2403.13188v1) | arXiv 2024 | I ∝ η(R)·I_e·ρ·cos α / R²· **near-range effect η(R) «significantly impacts the intensity data at closer ranges (R < 12 m)» λόγω lens defocusing**· Ouster reflectivity = βαθμονομημένο, «signal» = raw. | Άμεσα σχετικό: στο κοντινό δάπεδο (κάτω από τον αισθητήρα) το intensity κυριαρχείται από η(R) και α, που είναι **σταθερά ως προς τον αισθητήρα** → μοτίβο που ταξιδεύει μαζί του. |
| COIN-LIO (ICRA 2024), βλ. Ε1 | — | «brightness level varies smoothly throughout the image» λόγω απόστασης/γωνίας → brightness map σε μεγάλο παράθυρο + φίλτρο οριζόντιων γραμμών (ανομοιόμορφα elevation beams). | Παρόμοιο πρόβλημα, λύση με φιλτράρισμα εικόνας· δεν αναφέρει ρητά το δάπεδο ως πηγή ψευδο-στατικών features. |
| Du & Beltrame (ICRA 2023), βλ. Ε1 | — | ORB σε Ouster 1024×64· δεν αναφέρει φίλτρο δαπέδου. | — |

**Σύνοψη Ε4.** Η εξάρτηση του intensity από απόσταση/γωνία και το near-range effect είναι καλά
τεκμηριωμένα (Kashani 2015, Jutzi & Gross 2009, Viswanath 2024). Το COIN-LIO αντιμετωπίζει τη
συνέπεια στην εικόνα με φιλτράρισμα. **Δεν βρέθηκε** εργασία που να ονομάζει ρητά το πρόβλημα
«σταθερά ως προς τον αισθητήρα μοτίβα intensity στο κοντινό δάπεδο δίνουν αντιστοιχίσεις μηδενικής
κίνησης» και να το φιλτράρει γεωμετρικά (κριτήριο: δεν κινείται σχετικά με τον αισθητήρα). Το
φυσικό αίτιο όμως (η(R)·cos α / R² με σταθερή γεωμετρία δαπέδου–αισθητήρα) εξηγείται από τη
βιβλιογραφία και μπορεί να επικαλεστεί.

---

## Ε5. Oxford Spires — baselines στις ακολουθίες μας

Tao, Muñoz-Bañón, Zhang, Wang, Fu, Fallon — *The Oxford Spires Dataset* — [arXiv 2411.10546](https://ar5iv.labs.arxiv.org/html/2411.10546), IJRR 2025, [DOI 10.1177/02783649251369905](https://doi.org/10.1177/02783649251369905).
Αισθητήρας: Hesai QT64 (64 beams, 104° FoV, 10 Hz), 3 fisheye κάμερες, IMU κινητού 400 Hz, handheld «Frontier».

**Table 2 του paper (ATE, m) — μόνο οι ακολουθίες μας.** Όλα τα online baselines είναι **LiDAR-inertial** (με IMU)· **KISS-ICP δεν αξιολογείται** στο paper.

| Ακολουθία | Μήκος | VILENS-SLAM (LI) | Fast-LIO-SLAM (LI) | SC-LIO-SAM (LI) | ImMesh (LI) | HBA (offline) | COLMAP (offline, εικόνες) | **Δική μας (LiDAR-only)** | KISS-SLAM (δικό μας run) |
|---|---|---|---|---|---|---|---|---|---|
| christ-church-02 | 640 m | 0.17 | 0.49 | ✗ | 1.70 | 0.12 | 0.15 | **0.130** | — |
| christ-church-03 | 340 m | 0.03 | 0.23 | 0.14 | 0.16 | 0.05 | 0.07 | **0.038** | — |
| keble-college-02 | 290 m | 0.06 | 0.25 | 1.26 | 0.08 | 0.11 | 0.05 | **0.094** | 2.13 |

⚠️ Προσοχή στη σύγκριση: το paper δίνει ATE μετά από ευθυγράμμιση σε ολόκληρη την ακολουθία·
το δικό μας `church_02_cut.bag` είναι απόσπασμα 240 s / 2402 scans. Τα νούμερα είναι
συγκρίσιμα μόνο αν το πρωτόκολλο (`scripts/evaluate_gt.py`) ταιριάζει με το δικό τους
(ori-drs/oxford_spires_dataset). Να επαληθευτεί πριν από οποιαδήποτε δημοσίευση.

**KISS-ICP στο Oxford Spires (τρίτη πηγή).** Malladi et al. 2025 ([arXiv 2509.06593](https://arxiv.org/html/2509.06593v1)), Table I, ATE m / RPE %, **συγκεντρωτικά ανά τοποθεσία, χωρίς αριθμό ακολουθίας**:

| Μέθοδος | Blenheim | Bodleian | **Christ** | **Keble** | Radcliffe |
|---|---|---|---|---|---|
| KISS-ICP (LiDAR-only) | 1.16 / 46.99 | 2.15 / 19.10 | **0.96 / 11.01** | **9.48 / 38.48** | 0.46 / 3.58 |
| FAST-LIO2 | 0.94 / 1.11 | 0.26 / 0.26 | 0.72 / 23.64 | 6.06 / 5.42 | 0.46 / 1.40 |
| DLIO | 11.35 / 46.87 | 0.40 / 1.11 | 0.17 / 4.17 | 0.36 / 4.16 | 0.92 / 4.28 |
| Malladi et al. (LIO) | 0.20 / 0.95 | 0.86 / 0.74 | 0.25 / 0.89 | 0.08 / 0.96 | 0.15 / 0.70 |
| VILENS-SLAM | 0.56 / 1.13 | 1.11 / 1.68 | 0.11 / 0.38 | 0.12 / 0.39 | 0.07 / 0.39 |

Ανεξάρτητη επιβεβαίωση ότι το KISS-ICP (άρα και η οδομετρία του KISS-SLAM) υποαποδίδει σε
Keble/Christ Church, με την ίδια αιτιολογία που δίνουμε εμείς (μοντέλο σταθερής ταχύτητας).
Το KISS-SLAM ([arXiv 2503.12660](https://ar5iv.labs.arxiv.org/html/2503.12660), Guadagnino et al.
2025) χρησιμοποιεί την ίδια KISS-ICP οδομετρία («de-skewing … constant velocity motion model
prediction»)· αξιολογήθηκε σε MulRan, HeLiPR, Apollo, NCLT, Newer College — **όχι** Oxford Spires.

---

## Θέση της δικής μας μεθόδου

**Τι φαίνεται νέο (δεν βρέθηκε προηγούμενη αναφορά):**

1. **Εκτίμηση της κίνησης εντός scan LiDAR-only από χρονοσημασμένα ζεύγη 3D σημείων που
   προκύπτουν από SIFT σε πανόραμα intensity spinning LiDAR**, με RANSAC + robust προσαρμογή
   μοντέλου συνεχούς χρόνου. Οι υπάρχουσες intensity-image μέθοδοι για spinning LiDAR είτε
   αγνοούν την κίνηση εντός scan (MD-SLAM, Du & Beltrame, Guadagnino 2022) είτε την παίρνουν από
   IMU (RI-LIO, COIN-LIO). Οι continuous-time LiDAR-only μέθοδοι (CT-ICP, Traj-LO, ECTLO,
   STEAM-LO) την παίρνουν από **γεωμετρία**, όχι από intensity.
2. **Χρήση αυτής της κίνησης ταυτόχρονα ως deskew και ως αρχική θέση της ICP** σε
   KISS-ICP/KISS-SLAM, αντικαθιστώντας την πρόβλεψη από το προηγούμενο scan (και όχι ως
   επιπλέον residual σε φίλτρο, όπως COIN-LIO/RI-LIO).
3. **Η ποσοτική διάγνωση** ότι στο handheld Oxford Spires το CV deskew από το προηγούμενο scan
   είναι χειρότερο από καθόλου deskew — η βιβλιογραφία λέει «ανεπαρκές»/«μικρό κέρδος», όχι
   «ζημιά» (Ε3). Το ablation υπάρχει στο log: church_02 KISS 0.339 → χωρίς deskew 0.283 (#012)· keble-college-02
   2.133 → 0.183 (#032)· christ-church-03 0.122 → 0.125 (ίσο, #028). Ένα run ανά ακολουθία.
4. **Το γεωμετρικό φίλτρο ψευδο-στατικών αντιστοιχίσεων στο κοντινό δάπεδο** — το φυσικό αίτιο
   είναι γνωστό (near-range effect, cos α / R²), το συγκεκριμένο φίλτρο δεν βρέθηκε.

**Τι είναι γνωστό (πρέπει να αναφερθεί ως prior art):**

- Η ιδέα «εικόνα intensity lidar + χαρακτηριστικά + παρεμβολή θέσης στον χρόνο» υπάρχει από
  το 2012–2013 (Dong & Barfoot· Barfoot et al.), σε σαρωτικό lidar, χωρίς ICP — ⚠️ μερική
  επαλήθευση, να διαβαστούν τα πλήρη κείμενα.
- Lookup 3D σημείου μέσω index pixel εικόνας Ouster 64×1024 (Du & Beltrame 2023).
- Το intensity ως συμπληρωματική πληροφορία σε γεωμετρικά degenerate σκηνές (COIN-LIO,
  Intensity-SLAM, I-LOAM, RI-LIO).
- Ότι η υπόθεση σταθερής ταχύτητας του KISS-ICP αποτυγχάνει σε βίαιη/handheld/legged κίνηση
  (Stretch-ICP, Burnett 2024, Han & Lee 2026, Malladi 2025).
- Η εξάρτηση του intensity από R, α και η(R) (Kashani 2015, Jutzi & Gross 2009, Viswanath 2024).

**Ρητές αβεβαιότητες επαλήθευσης:** (α) Dong & Barfoot 2012 — μόνο μεταδεδομένα, όχι abstract/κείμενο·
(β) Guadagnino 2022, I-LOAM, Jutzi & Gross 2009 — μόνο abstract/μεταδεδομένα· (γ) RI-LIO deskew από
IMU — συμπέρασμα, όχι απόσπασμα· (δ) οι αριθμοί Oxford Spires Table 2 και Malladi Table I
προέκυψαν από αυτόματη εξαγωγή σελίδας — να διασταυρωθούν με το PDF πριν παρατεθούν σε κείμενο.
