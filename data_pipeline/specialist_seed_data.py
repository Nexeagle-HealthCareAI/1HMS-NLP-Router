# -*- coding: utf-8 -*-
"""
Hand-curated Hinglish symptom/keyword seed bank, one entry per specialist class.

Class list = EasyHMS's own canonical taxonomy (dbo.MedicalSpecialities.PatientFacingCategory,
deduplicated across the MD/MS/DM/MCh qualification ladder — see
easyHMSDatabase/db/data/seed/seed_medical_specialities.sql) PLUS Dentist and Veterinarian,
which the platform's existing router CSV already carried even though neither maps to that
NMC ladder (dentistry is a separate BDS/MDS track; veterinary isn't a human-hospital service
at all, but was kept per product decision).

Each class has three lists:
  symptoms         natural Hinglish sentences a patient might actually type/say, describing
                    how a symptom feels (varied length, word order, formality).
  doctor_mentions   short phrases that directly name the doctor/specialist type rather than
                    describing a symptom ("dil ka doctor", "skin specialist chahiye").
  keywords          bare symptom words/short phrases (not full sentences) — feeds word/char
                    n-gram signal without needing a full sentence template.

NOTE on classes that are structurally hard to reach from symptom text alone (kept anyway,
so the router doesn't 404 on these queries, but flagged so nobody's surprised by weaker
per-class accuracy):
  - Anaesthesiologist / Pathologist / Radiologist: patients don't self-route to these from
    a symptom — they're assigned during a care pathway (pre-op, lab draw, imaging). Seeded
    with the closest real patient-facing phrasing (procedure/report requests) instead of
    "symptoms".
  - GI/Surgical Gastroenterologist vs Gastroenterologist, Cardiothoracic Surgeon vs
    Cardiologist (Heart), Neurosurgeon vs Neurologist: a patient's own words essentially
    never distinguish "needs medicine" vs "needs surgery" for the same organ system — that's
    a triage call a GP/specialist makes after examination, not something inferable from a
    chief complaint. Seeded mainly with the minority of phrasings that DO carry a surgical
    signal (explicit mention of an operation, a lump/mass, trauma, or a named procedure);
    expect the model to lean on the broader medical-specialist sibling by default, with the
    surgical class rescued mostly by nearest-phrase search rather than the classifier.
"""

SEED = {

    # ── Existing 17 classes: broadened with more sentence shapes / vocabulary ──────────

    "Cardiologist (Heart)": {
        "keywords": ["seene mein dard", "chest pain", "dil ki dhadkan tez", "heart attack jaisa",
                     "saans phoolna", "BP high", "cholesterol zyada", "dil ghabrana"],
        "symptoms": [
            "Seene ke bich mein bhaari-pan sa mehsoos hota hai jab tez chalta hoon.",
            "Achanak dil bahut zor se dhadakne lagta hai, phir apne aap theek ho jata hai.",
            "Left arm mein ajeeb sa dard hota hai jo chest se shuru hota hai.",
            "Sidhi seedhi chadhte hi saans phoolne lagti hai aur chest tight lagta hai.",
            "Raat ko sote waqt achanak ghabrahat ho jaati hai aur dil tez chalta hai.",
            "Doctor ne bataya BP bahut high hai, sar bhi bhaari rehta hai.",
            "Thodi si mehnat karte hi thakaan aur chest mein jakadan feel hoti hai.",
          "Ankle aur pairon mein soojan aa gayi hai, saans bhi phool jaati hai lete waqt.",
            "Cholesterol report mein sab values high aayi hain, koi symptom nahi tha pehle.",
            "Chest mein jalan si hoti hai jo lagta hai gas nahi hai, kuch aur hai.",
            "Palpitations hoti hain, jaise dil ek beat miss kar deta hai.",
            "Family mein heart disease history hai, mujhe bhi checkup karana hai.",
            "Chalte chalte pairo mein dard hota hai jo rukne se turant theek ho jata hai.",
            "Bahut pasina aa raha tha aur chest mein bhari darad, ghabra gaya tha.",
        ],
        "doctor_mentions": ["dil ka doctor chahiye", "heart specialist se milna hai",
                             "cardiac doctor ko dikhana hai", "dil ki bimari ka daktar",
                             "heart checkup karwana hai"],
    },

    "Dermatologist (Skin)": {
        "keywords": ["skin par khujli", "daane nikal aaye", "rashes", "baal jhad rahe hain",
                     "skin allergy", "daag dhabbe", "acne", "fungal infection"],
        "symptoms": [
            "Poore body par laal daane nikal aaye hain jinme bahut khujli hoti hai.",
            "Face par acne aur pimples bahut badh gaye hain, marks bhi reh jaate hain.",
            "Skin bahut dry ho gayi hai aur jagah jagah patches pad gaye hain.",
            "Baal bahut zyada jhad rahe hain, kanghi karte waqt mutthi bhar aa jaate hain.",
            "Kisi cream ya soap se skin par turant rash ho jata hai.",
            "Naakhoon ka colour yellow ho gaya hai aur mota bhi ho gaya hai.",
            "Ghutno aur cheston par gol gol chakatte ban gaye hain, khujli bhi hoti hai.",
            "Underarms aur groin mein fungal infection jaisa lag raha hai, bahut khujli.",
            "Achanak poore chehre par sujan aa gayi allergy se lagta hai.",
            "Skin par white patches ban rahe hain jo dheere dheere fail rahe hain.",
            "Bohot zyada pasina aata hai aur usse skin mein irritation ho jaati hai.",
            "Muhaso ke baad skin par gehre daag reh jaate hain jo jaate nahi.",
        ],
        "doctor_mentions": ["skin specialist chahiye", "twacha rog ka doctor",
                             "dermatologist ko dikhana hai", "skin ka daktar"],
    },

    "General Physician": {
        "keywords": ["bukhar", "fever", "sardi zukam", "weakness", "body pain", "thakaan",
                     "ulti dast", "flu jaisa"],
        "symptoms": [
            "Do din se halka halka bukhar aa raha hai, sath mein badan dard bhi hai.",
            "Sardi, khansi aur zukam ho gaya hai, gale mein bhi kharash hai.",
            "Bahut kamzori aur thakaan mehsoos ho rahi hai bina kisi wajah ke.",
            "Poora din chakkar aa rahe hain aur kuch khaane ka mann nahi karta.",
          "Mausam badalte hi mujhe hamesha bukhar aur badan dard ho jata hai.",
            "General checkup karana hai, kaafi dino se theek nahi feel ho raha.",
            "Halka fever ke saath thodi si khansi bhi hai, do din se hai.",
            "Bas overall weakness hai, koi specific dard nahi hai kahin.",
            "Ghar mein sabko sardi zukam ho raha hai, mujhe bhi shuru ho gaya.",
            "Thoda sa bukhar hai aur bhookh bhi nahi lag rahi.",
        ],
        "doctor_mentions": ["general physician chahiye", "family doctor ko dikhana hai",
                             "normal doctor se checkup karana hai", "MBBS doctor chahiye"],
    },

    "Paediatrician": {
        "keywords": ["bacche ko bukhar", "baby ko ulti", "bachhe ka weight nahi badh raha",
                     "infant ko rashes", "bachhe ko khansi"],
        "symptoms": [
            "Mere bacche ko do din se tez bukhar aa raha hai, khana bhi nahi kha raha.",
            "6 mahine ke baby ko baar baar ulti ho rahi hai feed ke baad.",
            "Bachhe ka weight pichle kai mahino se badh hi nahi raha.",
            "Toddler ko raat ko neend nahi aati aur bahut rota rehta hai.",
            "Bachhe ke gale mein infection lag gaya lagta hai, dood peene se mana kar raha.",
            "Newborn ki skin peeli lag rahi hai, doctor ko dikhana zaroori hai kya.",
            "Bachhe ko vaccination ka schedule follow karna hai agla wala.",
            "Meri beti ko school mein baar baar pet dard hota hai subah ke time.",
            "Bacha bahut zyada rota hai aur pet fulaa hua lagta hai.",
            "Bachhe ko khansi ke saath ghrr ghrr ki awaaz aati hai saans lete waqt.",
        ],
        "doctor_mentions": ["child specialist chahiye", "bachho ka doctor dikhana hai",
                             "paediatrician se appointment chahiye", "baby doctor ko dikhana hai"],
    },

    "Psychiatrist": {
        "keywords": ["mood off rehta hai", "neend nahi aati", "anxiety", "akela mehsoos hota hai",
                     "negative thoughts", "depression jaisa lagta hai"],
        "symptoms": [
            "Aajkal mann bahut udaas rehta hai, kisi kaam mein mann nahi lagta.",
            "Raaton ko neend hi nahi aati, dimaag mein bohot vichar chalte rehte hain.",
            "Bahut zyada ghabrahat hoti hai bina kisi wajah ke, dil tez chalne lagta hai achanak.",
            "Kisi se baat karne ka mann nahi karta, akela rehna acha lagta hai.",
            "Chhoti chhoti baaton par bahut gussa aa jata hai aajkal.",
            "Bhookh bilkul nahi lagti aur weight bhi kam ho gaya hai stress se.",
            "Mind mein baar baar wahi negative khayal ghoomte rehte hain.",
            "Office ke stress ki wajah se panic attacks aane lage hain.",
            "Kabhi bahut high energy feel hoti hai, kabhi ekdum down ho jaata hoon.",
            "Kisi cheez mein interest nahi bacha, hamesha thaka thaka lagta hai mentally.",
        ],
        # NOTE: "dimaagi doctor" (mind/mental-state framing) is the established Psychiatrist
        # phrasing in the legacy dataset — deliberately NOT "dimaag ka doctor" (brain-organ
        # framing), which the legacy dataset already assigns to Neurologist. Keeping both
        # distinct avoids training the exact same phrase under two different labels.
        "doctor_mentions": ["mental health doctor chahiye", "psychiatrist se milna hai",
                             "dimaagi doctor", "counselor ya therapist chahiye",
                             "mansik rog ka doctor"],
    },

    "Neurologist": {
        "keywords": ["sar mein tez dard", "migraine", "chakkar aana", "haath pair sunn",
                     "seizure", "yaadasht kamzor"],
        "symptoms": [
            "Bahut tez sar dard hota hai, aankho ke aage roshni dikhti hai pehle.",
            "Achanak haath ya pair sunn ho jata hai kuch der ke liye.",
            "Chalte chalte achanak chakkar aa gaya aur balance bigad gaya.",
            "Yaadasht kamzor hoti ja rahi hai, chhoti baatein bhool jaate hain.",
            "Ek baar achanak behosh ho gaye the aur body akad gayi thi.",
            "Haath kaanpte hain jab kuch pakadne ki koshish karta hoon.",
            "Face ka ek side thoda tedha lag raha hai achanak se.",
            "Migraine ka dard hafte mein 2-3 baar ho jata hai, ulti bhi hoti hai saath.",
            "Neend mein achanak jhatke se uth jata hoon, poora badan kaanpta hai.",
            "Bolne mein ladkhadahat ho rahi hai achanak se, pehle theek tha.",
        ],
        "doctor_mentions": ["neurologist ko dikhana hai", "dimaag aur nervous system ka doctor",
                             "brain specialist chahiye"],
    },

    "Gastroenterologist": {
        "keywords": ["pet mein dard", "acidity", "gas bharna", "loose motion", "constipation",
                     "khatta dakaar"],
        "symptoms": [
            "Pet mein bahut dard hota hai khane ke turant baad.",
            "Khatte dakaar aate hain aur seene mein jalan hoti hai roz.",
            "Kai din se pet saaf nahi ho raha, bahut takleef hoti hai.",
            "Loose motions lag gaye hain do din se, pet mein murmurahat bhi hai.",
            "Khana khate hi pet fool jaata hai, gas bahut banti hai.",
            "Stool mein kabhi kabhi blood dikh raha hai, dar lag raha hai.",
            "Bina kisi wajah ke pichle mahine se weight kam ho raha hai, bhookh bhi kam hai.",
            "Pet ke upar wale hisse mein jalan si rehti hai khaali pet.",
            "Bahut zyada belching hoti hai aur pet bhara bhara lagta hai hardam.",
            "Rat ko sote waqt khatta pani gale tak aa jata hai.",
        ],
        "doctor_mentions": ["pet ka doctor chahiye", "gastro doctor ko dikhana hai",
                             "stomach specialist se milna hai"],
    },

    "Orthopaedic Surgeon (Bone)": {
        "keywords": ["ghutno mein dard", "kamar dard", "joint pain", "haddi mein chot",
                     "fracture", "peeth dard"],
        "symptoms": [
            "Ghutno mein uthte baithte bahut dard hota hai, seedhi chadhna mushkil hai.",
            "Kamar mein bahut dard hai, jhukne mein bhi dikkat hoti hai.",
            "Girne ke baad haath mein bahut dard hai, lagta hai fracture ho gaya hai.",
            "Kandhe ko upar uthane mein dard hota hai, raat ko bhi so nahi paata.",
            "Subah uthte hi joints akde hue lagte hain, dheere dheere theek hota hai.",
            "Ghutno se chatak chatak ki awaaz aati hai chalte waqt.",
            "Gardan mudne mein dard hota hai, kaafi din se hai ye problem.",
            "Ankle mudh gaya khelte waqt, sujan bhi aa gayi hai turant.",
            "Kaafi der khade rehne se ediyon mein bahut dard hone lagta hai.",
            "Peeth ke nichle hisse mein dard hai jo pair tak jata hai kabhi kabhi.",
        ],
        "doctor_mentions": ["haddi ka doctor chahiye", "bone specialist ko dikhana hai",
                             "orthopaedic surgeon se milna hai", "joint ka doctor"],
    },

    "Ophthalmologist (Eye)": {
        "keywords": ["aankhon mein jalan", "dhundla dikhna", "aankh laal", "aankhon mein khujli",
                     "chashma", "aankho se pani"],
        "symptoms": [
            "Aankhon mein bahut jalan hoti hai screen dekhte dekhte.",
            "Door ki cheezein dheere dheere dhundli dikhne lagi hain.",
            "Aankh achanak laal ho gayi hai aur usme se pani aa raha hai.",
            "Raat ko gaadi chalate waqt saamne se aane wali lights se aankhein chundh jaati hain.",
            "Aankhon mein bahut khujli hoti hai aur kabhi kabhi sujan bhi aa jaati hai.",
            "Kisi cheez ko paas se dekhne mein bhi dikkat ho rahi hai aajkal.",
            "Aankh mein kuch chubhta hua sa lagta hai, dekhne mein kirkiri jaisi feel hoti hai.",
            "Chashma ka number badalna hai shayad, dikhna kam ho gaya hai.",
            "Aankho ke saamne kabhi kabhi kaale dhabbe se tair jaate hain.",
            "Subah uthte hi aankhein chipak jaati hain, pili sa discharge bhi hota hai.",
        ],
        "doctor_mentions": ["eye specialist chahiye", "aankho ka doctor dikhana hai",
                             "ophthalmologist se milna hai", "chashme wala doctor"],
    },

    "ENT Specialist": {
        "keywords": ["gale mein kharash", "kaan mein dard", "naak band", "awaaz baithna",
                     "sinus", "gale mein kuch atka"],
        "symptoms": [
            "Gale mein bahut kharash hai aur nigalte waqt dard hota hai.",
            "Kaan mein dard ho raha hai aur sunai bhi kam de raha hai.",
            "Naak hamesha band rehti hai, saans lene mein dikkat hoti hai.",
            "Awaaz baith gayi hai kai dino se, bolne mein museebat ho rahi hai.",
            "Kaan mein se ajeeb si awaaz aati rehti hai, ghanti jaisi.",
            "Baar baar sinus ka dard hota hai, matha bhi bhaari rehta hai.",
            "Gale mein kuch atka hua sa feel hota hai hardam.",
            "Naak se khoon aa gaya tha achanak, koi chot nahi lagi thi.",
            "Tonsils bahut soojh gaye hain, khana nigalna mushkil ho raha hai.",
            "Kaan mein khujli aur halka dard rehta hai, paani gaya tha shayad.",
        ],
        "doctor_mentions": ["ENT doctor chahiye", "gala kaan naak ka doctor",
                             "throat specialist ko dikhana hai"],
    },

    "Endocrinologist (Hormones/Diabetes)": {
        "keywords": ["sugar high", "diabetes", "thyroid", "weight badhna ghatna", "hormone imbalance",
                     "baar baar pyaas lagna"],
        "symptoms": [
            "Sugar level check karvaya to bahut high aaya, pehle kabhi nahi tha.",
            "Baar baar pyaas lagti hai aur baar baar washroom bhi jaana padta hai.",
            "Bina kuch kiye weight bahut badh gaya hai pichle kuch mahino mein.",
            "Thyroid ki problem hai shayad, gale mein sujan bhi lagti hai halki.",
            "Bahut thakaan rehti hai aur mood bhi low rehta hai, periods bhi irregular hain.",
            "Haath pair mein jhanjhanahat rehti hai, doctor ne sugar check karne bola.",
            "Achanak bahut zyada bhookh lagne lagi hai, weight kam ho raha hai phir bhi.",
            "Family mein diabetes hai, mujhe bhi apna sugar check karana hai.",
            "Weight loss ke saath saath dil bhi tez dhadakta hai, garmi bhi zyada lagti hai.",
        ],
        "doctor_mentions": ["hormone specialist chahiye", "sugar ka doctor dikhana hai",
                             "diabetes specialist se milna hai", "thyroid doctor chahiye"],
    },

    "Gynaecologist": {
        "keywords": ["periods irregular", "pregnancy checkup", "pet ke niche dard", "white discharge",
                     "PCOD", "menstrual cramps"],
        "symptoms": [
            "Periods bahut irregular ho gaye hain pichle kai mahino se.",
            "Pregnancy confirm karani hai, periods miss ho gaye hain.",
            "Pet ke niche wale hisse mein periods ke doraan bahut dard hota hai.",
            "White discharge zyada ho raha hai, thoda irritation bhi hai.",
            "PCOD ka doubt hai, weight bhi badh raha hai aur periods bhi skip hote hain.",
            "Pregnancy ke doraan checkup karana hai regular, 5th month chal raha hai.",
            "Bahut heavy bleeding hoti hai periods ke time, pehle aisa nahi tha.",
            "Menopause ke baad se hot flashes aur mood swings ho rahe hain.",
            "Breast mein ek gaanth si mehsoos hui hai, dar lag raha hai.",
        ],
        "doctor_mentions": ["lady doctor chahiye", "gynaecologist se milna hai",
                             "mahila rog vishesagya", "obstetrician ko dikhana hai"],
    },

    "General Surgeon": {
        # NOTE: bare "surgeon chahiye" / "surgery chahiye" deliberately excluded — now that
        # 6 surgical-sibling classes exist (Vascular/Cardiothoracic/GI-Surgical/Neuro/Plastic/
        # General), an unqualified "I need a surgeon" genuinely doesn't signal which one; forcing
        # it to General Surgeon just teaches the model a wrong-often guess. Only phrases that
        # name a hernia/appendix/gallbladder/general-surgery-specific context are kept here.
        "keywords": ["operation karana hai", "hernia", "gaanth", "appendix", "chot gehri hai"],
        "symptoms": [
            "Pet ke niche ek gaanth ban gayi hai jo khaansne par bahar aati hai, hernia lagta hai.",
            "Achanak pet ke right side mein bahut tez dard hua, appendix ho sakta hai.",
            "Gallbladder mein stone hai, doctor ne operation ki salah di hai.",
            "Skin ke neeche ek gaanth hai jo dheere dheere badh rahi hai.",
            "Deep cut lag gaya hai, stitches lagane padenge shayad.",
            "Piles ki problem hai kaafi time se, ab operation ki zaroorat bata rahe hain.",
            "Body mein kahin bhi lump feel ho raha hai jo pehle nahi tha.",
            "Operation ke baad follow-up checkup karana hai ghaav dekhne ke liye.",
        ],
        "doctor_mentions": ["operation wale doctor se milna hai",
                             "general surgery specialist ko dikhana hai",
                             "general surgeon ko dikhana hai"],
    },

    "Pulmonologist (Chest/Lungs)": {
        "keywords": ["saans lene mein dikkat", "khansi lagatar", "asthma", "chest congestion",
                     "wheezing", "TB ka doubt"],
        "symptoms": [
            "Kaafi hafton se khansi lagatar bani hui hai, jaati hi nahi.",
            "Saans lene mein bahut dikkat hoti hai, seeti jaisi awaaz bhi aati hai.",
            "Asthma ka attack aane wala lagta hai, chest tight ho gaya hai.",
            "Khansi ke saath khoon bhi aa raha hai thoda thoda.",
            "Raat ko sote waqt saans phool jaati hai achanak.",
            "Chest mein bahut congestion feel hota hai, balgam nikal nahi raha.",
            "Thodi si mehnat mein hi saans phool jaati hai kaafi dino se.",
            "TB ka doubt hai, weight bhi kam ho raha hai aur raat ko paseena aata hai.",
        ],
        "doctor_mentions": ["chest specialist chahiye", "lungs ka doctor dikhana hai",
                             "pulmonologist se milna hai", "saans ka doctor"],
    },

    "Urologist": {
        "keywords": ["peshaab mein jalan", "kidney stone", "peshaab baar baar", "prostate",
                     "peshaab mein khoon"],
        "symptoms": [
            "Peshaab karte waqt bahut jalan hoti hai kai dino se.",
            "Peshaab mein khoon dikha aaj, dar lag raha hai.",
            "Baar baar peshaab jaana padta hai, raat ko bhi kai baar uthna padta hai.",
            "Kamar ke ek side mein achanak bahut tez dard hua, kidney stone lagta hai.",
            "Peshaab ruk ruk kar aata hai aur poora khali bhi nahi hota lagta.",
            "Prostate badh gaya hai lagta hai, peshaab ka flow kamzor ho gaya hai.",
            "Peshaab mein badbu aur jhaag zyada aa raha hai aajkal.",
        ],
        "doctor_mentions": ["urologist ko dikhana hai", "kidney aur peshaab ka doctor",
                             "prostate specialist chahiye"],
    },

    # ── New classes (were entirely missing from the old 17-class CSV) ─────────────────

    "Anaesthesiologist": {
        "keywords": ["anesthesia", "operation se pehle", "behosh karna", "pre-op consultation",
                     "epidural", "spinal block"],
        "symptoms": [
            "Operation se pehle anesthesia ke baare mein doctor se baat karni hai.",
            "Pichli baar anesthesia se ulti hui thi, is baar precaution leni hai.",
            "Surgery ke doraan behosh karne wale doctor se milna hai pehle.",
            "Anesthesia ka asar utarne mein bahut time lagta hai mujhe, batana hai unko.",
            "Delivery ke liye epidural lena hai ya nahi, samajh nahi aa raha.",
            "Kamar mein spinal anesthesia dena hai operation ke liye, doubts hain kuch.",
            "Anesthesia se allergy hai kya check karna hai operation se pehle.",
            "Local anesthesia mein hi chhota operation ho sakta hai kya poochna hai.",
            "Pain management ke liye anesthesia specialist se consultation chahiye.",
            "General anesthesia diya jayega ya sirf sunn karenge, ye janna hai operation se pehle.",
            "Meri maa ko heart ki problem hai, unhe anesthesia dena safe hoga kya operation mein.",
            "Bache ko chhote operation ke liye anesthesia denge, kitna risk hota hai puchna hai.",
            "Anesthesia lene ke kitne ghante pehle khana peena band karna hai.",
            "Operation ke baad dard kam karne ke liye kaunsi anesthesia use hogi, discuss karna hai.",
            "Pichli surgery mein anesthesia se saans lene mein dikkat hui thi, is baar dar lag raha hai.",
        ],
        "doctor_mentions": ["anesthesia doctor se milna hai", "anaesthesiologist ko dikhana hai",
                             "behoshi wale doctor se baat karni hai", "pain management specialist chahiye",
                             "operation se pehle anesthesia consult chahiye"],
    },

    "Emergency Medicine Specialist": {
        "keywords": ["accident ho gaya", "bahut khoon beh raha hai", "behosh ho gaye",
                     "emergency hai", "zeher kha liya"],
        "symptoms": [
            "Accident ho gaya hai, bahut khoon beh raha hai turant madad chahiye.",
            "Ghar mein koi achanak behosh ho gaya hai, saans bhi theek se nahi le pa raha.",
            "Bachhe ne galti se kuch dawai kha li hai, turant dikhana hai.",
            "Seene mein achanak bahut tez dard hua hai abhi, ambulance chahiye.",
            "Seedhi se girne ke baad hil nahi pa raha, turant hospital jaana hai.",
            "Jalne ki wajah se skin poori tarah se badly burn ho gayi hai.",
            "Saans achanak ruk si gayi thi, neela pad gaya tha chehra.",
            "Snake ne kaat liya hai, turant hospital pahunchna hai.",
            "High fever ke saath bachhe ko seizure aa gaya achanak.",
            "Kisi ne galti se poison pi liya hai, turant hospital chahiye.",
            "Gaadi ka accident hua hai, kai log ghayal hain, turant madad chahiye.",
            "Chhat se gir gaya bachha, sar se khoon beh raha hai turant.",
            "Achanak chest jakad gaya aur saans bilkul nahi le pa rahe the.",
            "Bahut zyada allergic reaction ho gaya khane se, gala sujj raha hai turant.",
            "Deep cut lag gaya hai kitchen mein, khoon rukk nahi raha, turant dikhana hai.",
            "Bijli ka jhatka laga hai kisi ko, behosh pade hain abhi.",
            "Kisi ki heat stroke se tabiyat achanak bigad gayi hai bahar dhoop mein.",
        ],
        "doctor_mentions": ["emergency mein turant dikhana hai", "casualty le jaana hai",
                             "turant doctor chahiye abhi", "ambulance bulani hai turant",
                             "24x7 emergency ward jaana hai"],
    },

    "Geriatrician": {
        "keywords": ["budhape ki kamzori", "senior citizen ki tabiyat", "budhon ki bimari",
                     "buzurg ko chakkar"],
        "symptoms": [
            "Mere dada ji ko chalte waqt balance nahi banta, gir jaane ka dar rehta hai.",
            "Nani ji ko har cheez bhoolne lagi hai, ye normal budhapa hai ya bimari.",
            "Senior citizen hain, kai dawaiyan chal rahi hain, overall checkup karana hai.",
            "Buzurg papa ko uthne baithne mein bahut dikkat hoti hai aajkal.",
            "Age zyada hai unki, appetite bhi kam ho gayi hai aur weakness rehti hai.",
            "80 saal ke hain, akele rehte hain, unka regular health checkup karana hai.",
            "Buzurg maa ko raat ko confusion hoti hai, din mein theek rehti hain.",
            "Dadi ji gir gayi thi ghar mein, ab chalne mein bahut dar lagta hai unhe.",
            "Umar ke saath unhe kai bimariyan ho gayi hain, sabka ek saath dekhna hai.",
            "75 saal ke papa ko subah subah bahut chakkar aate hain uthte waqt.",
            "Buzurg ammi kaafi weak ho gayi hain, khana bhi kam kha rahi hain aajkal.",
            "Ghar ke buzurgo ke liye regular full body checkup karwana chahte hain.",
            "Unki kai dawaiyan chal rahi hain alag alag doctor ki, ek saath review karana hai.",
            "Papa ko neend bahut kam aati hai raat mein, din mein sote rehte hain.",
            "Buzurg dadi ko akele chhodna theek nahi lagta, ghar par hi doctor bula sakte hain kya.",
        ],
        "doctor_mentions": ["buzurgo ka doctor chahiye", "geriatric specialist ko dikhana hai",
                             "senior citizen specialist se milna hai", "budhape ki bimariyon ka doctor"],
    },

    "Pathologist": {
        "keywords": ["blood report", "lab test", "sample dena hai", "report samajhni hai",
                     "culture test", "histopathology"],
        "symptoms": [
            "Meri blood report aayi hai, samajh nahi aa raha values ka matlab kya hai.",
            "Fasting blood sample dena hai kal subah, kahan jama karna hai.",
            "Biopsy report abhi tak nahi aayi, kab tak milegi.",
            "Lab test ke results normal range se bahar aaye hain, doctor ko dikhana hai.",
            "Urine culture test karwana hai infection check karne ke liye.",
            "Histopathology report samajhni hai, doctor ne bola tha lab se le lo.",
            "Stool sample test karwana hai, kaha jama karein.",
            "CBC report mein kuch values low aayi hain, matlab samajhna hai.",
            "PCR test karwana hai, kitne din mein report milegi.",
            "Thyroid ka blood test karwana hai fasting mein ya normal.",
            "Report mein ek value red mark ki hai, kitna serious hai janna hai.",
            "Sugar test PP aur fasting dono karwane hain, ek hi din mein ho jayenge kya.",
            "Allergy test karwana hai, kaunse kaunse cheezon se allergy hai pata karna hai.",
            "Pregnancy test karwana hai lab mein, ghar wala test positive aaya tha.",
        ],
        "doctor_mentions": ["lab report checkup", "pathologist se report discuss karni hai",
                             "sample test karwana hai", "lab wale doctor se baat karni hai",
                             "blood test report samajhni hai"],
    },

    "Physiotherapist / Rehab": {
        "keywords": ["exercise se dard theek", "physiotherapy chahiye", "movement kam ho gaya",
                     "operation ke baad rehab"],
        "symptoms": [
            "Operation ke baad se ghutna hilana mushkil hai, exercises karni hain shayad.",
            "Kandha jam sa gaya hai, movement bahut kam ho gayi hai.",
            "Doctor ne bola tha physiotherapy karo par kaafi time se nahi ki.",
            "Stroke ke baad ek side kamzor ho gayi hai, chalna practice karna hai.",
            "Kamar ke dard mein exercise se hi aaram milta hai, dawai se nahi.",
            "Sports injury ke baad muscle strength wapas laani hai dheere dheere.",
            "Accident ke baad chalna phirna bhool gaye hain, dobara sikhna hai.",
            "Frozen shoulder hai bola doctor ne, exercises se hi thik hoga.",
            "Paralysis ke baad ek haath bilkul kaam nahi kar raha, therapy chahiye.",
            "Knee replacement ke baad chalna phir se sikhna hai proper tarike se.",
            "Bache ko cerebral palsy hai, uski movement therapy chalti rehti hai.",
            "Pairo mein weakness hai polio ke baad se, exercises se improve hota hai kya.",
            "Ghar par hi physiotherapy karwa sakte hain kya, aana jaana mushkil hai.",
            "Neck ka dard hai desk job ki wajah se, posture theek karne wali exercises chahiye.",
        ],
        "doctor_mentions": ["physiotherapist chahiye", "rehab specialist se milna hai",
                             "exercise therapy wale doctor", "physio se appointment chahiye",
                             "movement therapy specialist chahiye"],
    },

    "Oncologist (Cancer)": {
        "keywords": ["cancer ka doubt", "gaanth badh rahi hai", "biopsy mein cancer",
                     "keemo therapy", "tumor"],
        "symptoms": [
            "Breast mein gaanth thi, biopsy mein cancer confirm hua hai.",
            "Achanak bahut weight kam ho raha hai bina kisi wajah ke, cancer ka dar hai.",
            "Gaanth dheere dheere badh rahi hai aur ab dard bhi hone laga hai.",
            "Keemo therapy shuru karni hai, kahan aur kaise hoti hai janna hai.",
            "Scan mein ek tumor dikha hai, aage kya karna hai samajh nahi aa raha.",
            "Baar baar mooh mein ulcer hota hai jo theek hi nahi hota, dar lag raha hai.",
            "Gale mein ek ganth hai jo kam nahi ho rahi kai hafton se.",
            "Radiation therapy shuru karni hai, process samajhni hai poori.",
            "Second opinion chahiye cancer treatment ke liye, report bhej sakte hain.",
            "Skin par ek mole ka shape aur colour badal gaya hai achanak.",
            "Papa ko lung cancer detect hua hai, aage ka treatment plan samajhna hai.",
            "Keemo ke side effects bahut zyada ho rahe hain, kuch karna hai iske liye.",
            "Family mein cancer history hai, screening karwani hai preventive.",
            "Ek hi jagah baar baar khoon aata hai bina kisi chot ke, dar lag raha hai.",
            "Cancer stage kya hai aur survival rate kya hoga, ye janna hai.",
        ],
        "doctor_mentions": ["cancer specialist chahiye", "oncologist se milna hai",
                             "cancer ka doctor dikhana hai", "tumor specialist se milna hai",
                             "keemo therapy center ka pata chahiye"],
    },

    "Radiologist": {
        "keywords": ["MRI karwani hai", "X-ray karwana hai", "CT scan", "ultrasound", "sonography",
                     "mammography", "bone density test"],
        "symptoms": [
            "Doctor ne MRI karwane ko bola hai, kahan achi facility milegi.",
            "Haath mein fracture lag raha hai, X-ray karwana hai turant.",
            "Pregnancy ka sonography karwana hai is mahine ka.",
            "CT scan ki report samajhni hai, kya nikla hai usme.",
            "Ultrasound mein kuch clear nahi dikha, dobara karwana padega kya.",
            "Mammography karwani hai routine checkup ke liye.",
            "Doppler test karwana hai pairo ki naso ka, doctor ne bola hai.",
            "Bone density test karwana hai, budhape mein haddi kamzor ho rahi hai.",
            "PET scan karwana hai, doctor ne cancer follow-up ke liye bola hai.",
            "Ghutne ka MRI karwana hai ligament check karne ke liye.",
            "Bache ke fracture ka X-ray dobara karwana hai healing check karne.",
            "Echo test karwana hai heart ka, doctor ne bataya hai.",
            "Brain ka MRI karwana hai, sar dard bahut zyada rehta hai.",
            "Report mein kya nikla hai, radiologist se khud baat kar sakte hain kya.",
        ],
        "doctor_mentions": ["scan karwana hai", "radiologist se report discuss karni hai",
                             "imaging center jana hai", "MRI center ka pata chahiye",
                             "scan report samajhni hai"],
    },

    "Sports Medicine Specialist": {
        "keywords": ["khelte waqt chot", "gym mein chot lagi", "muscle pull", "ligament injury",
                     "sports injury"],
        "symptoms": [
            "Football khelte waqt ghutna mud gaya, ab chalna mushkil ho raha hai.",
            "Gym mein weight uthate waqt kandhe mein kuch chatak gaya jaisa laga.",
            "Running karte waqt ankle twist ho gaya, sujan bhi aa gayi hai.",
            "Cricket khelte waqt haath mein ball lag gayi thi, ab bhi dard hai.",
            "Muscle pull ho gaya hai practice ke doraan, kal se bahut dard hai.",
            "Ligament mein chot lagi thi match ke doraan, ab tak poora recover nahi hua.",
            "Marathon practice ke baad ghutno mein bahut dard rehta hai roz.",
            "Gym mein injury ke baad wapas training start karni hai safely.",
            "Badminton khelte waqt kalai mud gayi thi, ab bhi weak lagti hai.",
            "Swimming karte waqt kandhe mein dard shuru ho gaya hai roz roz.",
            "Trekking ke baad pairo mein bahut soojan aur dard hai kai din se.",
            "College tournament ke liye jaldi recover hona hai, injury management chahiye.",
            "Weightlifting karte waqt kamar mein kuch kadak gaya jaisa laga.",
        ],
        "doctor_mentions": ["sports injury specialist chahiye", "sports medicine doctor ko dikhana hai",
                             "athlete injury specialist chahiye", "fitness aur injury doctor chahiye"],
    },

    "Rheumatologist": {
        "keywords": ["joints mein soojan", "arthritis", "subah akadan", "auto-immune",
                     "joint pain multiple"],
        "symptoms": [
            "Kai joints mein ek saath soojan aur dard rehta hai kaafi dino se.",
            "Subah uthte hi ek ghante tak joints akde hue lagte hain.",
            "Arthritis ka doubt hai, ungliyon ke joints mein bhi dard hone laga hai.",
            "Dono haathon ke chhote joints mein symmetrical dard rehta hai.",
            "Doctor ne bola auto-immune ho sakta hai, joints ke saath skin par bhi rash hai.",
            "Bahut thakaan rehti hai aur haath ki ungliyan subah akdi hui lagti hain.",
            "Joints mein dard ke saath saath halka bukhar bhi rehta hai kaafi dino se.",
            "Ghutno ke saath kalai aur ungliyon mein bhi soojan aa gayi hai.",
            "Lupus ka doubt hai, joints ke saath chehre par bhi rash hai dhoop mein.",
            "Gout ki problem hai, pair ke angoothe mein achanak bahut tez dard hota hai.",
            "Kandhe aur kamar dono mein akadan rehti hai, movement kam ho gayi hai.",
            "Doctor ne blood test mein RA factor check karne ko bola hai.",
        ],
        "doctor_mentions": ["rheumatologist ko dikhana hai", "joint aur auto-immune specialist chahiye",
                             "arthritis specialist se milna hai", "gout specialist se milna hai"],
    },

    "Nephrologist (Kidney)": {
        "keywords": ["kidney ki problem", "creatinine high", "chehre par sujan", "kidney failure",
                     "dialysis"],
        "symptoms": [
            "Blood test mein creatinine bahut high aaya hai, kidney specialist se milna hai.",
            "Subah uthte hi aankho ke aas paas aur pairo mein sujan rehti hai.",
            "Peshaab bahut kam ho gaya hai pichle kuch dino se.",
            "Doctor ne bataya kidney function kam ho gaya hai, dialysis ki baat kar rahe the.",
            "Bahut thakaan aur ulti si feel hoti hai, kidney test karwane bole hain.",
            "Peshaab mein bahut jhaag aata hai kaafi dino se, protein leak ho sakta hai.",
            "Dono kidney mein stone hai bola hai, ab specialist se milna hai.",
            "BP control nahi ho raha kai dawaiyon ke baad bhi, kidney check karana hai.",
            "Diabetes ke saath kidney ki reports bhi kharab aane lagi hain.",
            "Dialysis start karni padegi shayad, doctor ne bataya hai reports dekhkar.",
            "Chehre aur pairo mein subah subah zyada sujan rehti hai, sham tak kam ho jaati hai.",
            "Kidney transplant ke baare mein jaankari chahiye, doctor ne suggest kiya hai.",
        ],
        "doctor_mentions": ["kidney specialist chahiye", "nephrologist se milna hai",
                             "dialysis specialist ko dikhana hai", "kidney transplant specialist chahiye"],
    },

    "GI/Surgical Gastroenterologist": {
        "keywords": ["pet ki surgery", "gallbladder operation", "hernia operation pet",
                     "GI operation", "fistula", "rectal bleeding operation"],
        "symptoms": [
            "Gallbladder mein stones hain aur doctor ne operation karne ko bola hai.",
            "Pet mein ek gaanth hai jo operation se hi nikalni padegi bola hai.",
            "Piles bahut badh gaye hain, ab surgery ke alawa koi option nahi bacha.",
            "Endoscopy mein kuch nikla hai jiske liye operation ki zaroorat hai.",
            "Fistula ho gaya hai, doctor ne surgery karne ki salah di hai.",
            "Aant mein rukawat hai bola hai doctor ne, operation karna padega.",
            "Rectal bleeding bahut zyada ho rahi hai, surgeon ne dikhne ko bola hai.",
            "Liver mein cyst hai jise operation se nikalna padega bola hai.",
            "Hernia bahar dikhne laga hai khaansne par, operation ki zaroorat bata rahe hain.",
            "Pancreas mein problem hai, surgery karke hi theek hoga bola doctor ne.",
            "Anal fissure bahut dard de raha hai, surgery ke alawa option nahi bacha.",
            "Appendix ka operation karwana hai, doctor ne turant bola hai.",
        ],
        "doctor_mentions": ["GI surgeon se milna hai", "pet ki surgery wale doctor chahiye",
                             "gastro surgery specialist chahiye", "hernia operation specialist chahiye"],
    },

    "Vascular Surgeon": {
        "keywords": ["nason ka phoolna", "varicose veins", "pairo ki naso mein dard",
                     "blood clot pair mein"],
        "symptoms": [
            "Pairo ki nasein phool kar bahar dikhne lagi hain, khade rehne se dard hota hai.",
            "Pair mein achanak sujan aur dard hua hai, blood clot ka dar hai.",
            "Chalte waqt pairon mein dard hota hai, thodi der rukne se aaram milta hai.",
            "Ek pair doosre se zyada thanda rehta hai, colour bhi halka nila lagta hai.",
            "Pairo ki nasein neeli aur ubhri hui dikhti hain, dekhne mein bhi ajeeb lagti hain.",
            "Ghaav pair mein hai jo bahut dino se bhar hi nahi raha.",
            "Chalne mein bahut takleef hoti hai, doctor ne blood flow check karne bola.",
            "Diabetes hai aur pair ka ghaav thik nahi ho raha, kaala bhi pad raha hai jagah jagah.",
            "Gardan ki artery mein blockage bataya doctor ne, operation ki baat kar rahe hain.",
            "Varicose veins ki wajah se pairo mein bhaari-pan aur dard rehta hai poora din.",
        ],
        # NOTE: deliberately NOT the bare "naso ka doctor" — "nas"/"naso" is genuinely
        # ambiguous in colloquial Hinglish (nerve vs. blood vessel), and the legacy dataset
        # already established "Nason ka doctor" -> Neurologist. Only unambiguous, vein-specific
        # phrasing (varicose veins, blood vessel) goes here to avoid a same-text label conflict.
        "doctor_mentions": ["vascular surgeon ko dikhana hai", "varicose veins ka doctor",
                             "blood vessel specialist se milna hai", "varicose veins specialist chahiye"],
    },

    "Cardiothoracic Surgeon": {
        "keywords": ["heart surgery", "bypass surgery", "valve replace", "heart operation",
                     "congenital heart defect", "chest surgery lungs"],
        "symptoms": [
            "Doctor ne bataya heart ki ek valve kharab hai, operation ki baat kar rahe hain.",
            "Bypass surgery karwani hai, angiography mein blockage nikla hai.",
            "Heart operation ke baad recovery ke baare mein baat karni hai.",
            "Bache ko janam se dil mein ched hai, surgery karni padegi bola hai.",
            "Angioplasty se theek nahi hua, ab open heart surgery bata rahe hain.",
            "Lungs mein tumor hai jo operation se nikalna padega, chest surgeon chahiye.",
            "Heart transplant ke baare mein jaankari chahiye, doctor ne refer kiya hai.",
            "Chest ke andar fluid bhar gaya hai bola hai, operation se nikalna padega.",
            "Heart ki teeno arteries mein blockage hai, bypass ki taiyari karni hai.",
        ],
        "doctor_mentions": ["heart surgeon se milna hai", "cardiothoracic surgeon ko dikhana hai",
                             "chest aur heart surgery specialist chahiye", "bypass surgery specialist chahiye"],
    },

    "Neurosurgeon": {
        "keywords": ["brain tumor", "sar mein gehri chot", "spine operation", "dimaag ki surgery"],
        "symptoms": [
            "Scan mein brain mein ek growth dikha hai, operation ki zaroorat bata rahe hain.",
            "Sar mein bahut gehri chot lagi thi accident mein, behosh bhi ho gaye the.",
            "Spine mein disc slip ho gayi hai, doctor operation ki salah de rahe hain.",
            "Dimaag ki nas phatne jaisa laga tha achanak, turant hospital le gaye the.",
            "Kamar ke dard ke saath pair mein bhi kamzori aa gayi hai, disc ka issue lagta hai.",
            "Sar mein paani bhar gaya hai bola hai doctor ne, operation karna padega.",
            "Epilepsy control nahi ho raha dawaiyon se, surgery ka option poochna hai.",
            "Gardan ke pas ki spine mein pinch hui nas hai, haath mein weakness aa gayi hai.",
            "Accident mein sar mein blood clot ban gaya hai, turant operation bata rahe hain.",
        ],
        "doctor_mentions": ["neurosurgeon ko dikhana hai", "brain surgery specialist chahiye",
                             "spine surgery specialist se milna hai", "brain tumor specialist chahiye"],
    },

    "Plastic Surgeon": {
        "keywords": ["jalne ke nishan", "cosmetic surgery", "chehre ki surgery", "scar hatana"],
        "symptoms": [
            "Jalne ke bahut gehre nishan reh gaye hain, unhe theek karwana hai.",
            "Accident ke baad chehre par scar reh gaya hai, surgery se theek ho sakta hai kya.",
            "Cosmetic surgery ke baare mein consultation chahiye.",
            "Skin graft karwana hai jalne wali jagah par.",
            "Kaan ka shape thoda alag hai janam se, surgery se theek ho sakta hai kya.",
            "Mastectomy ke baad reconstruction ke baare mein baat karni hai.",
            "Haath ki ungli kat gayi thi accident mein, usse theek karwana hai.",
            "Nose ka shape thoda change karwana hai, cosmetic reasons se.",
            "Dog bite ke baad chehre par gehra ghaav hai, plastic surgery se theek hoga kya.",
            "Weight loss surgery ke baad skin loose ho gayi hai, tighten karwani hai.",
        ],
        "doctor_mentions": ["plastic surgeon ko dikhana hai", "cosmetic surgery specialist chahiye",
                             "reconstruction surgery ke baare mein baat karni hai", "skin graft specialist chahiye"],
    },

    # ── Kept as-is per product decision, despite not mapping to the NMC ladder ────────

    "Dentist": {
        "keywords": ["daant mein dard", "muh mein ulcer", "masudo se khoon", "daant hilna",
                     "cavity"],
        "symptoms": [
            "Daant mein bahut dard hai kai dino se, thanda garam lagta hai turant.",
            "Masudo se khoon aata hai brush karte waqt.",
            "Ek daant hilne laga hai bina kisi chot ke.",
            "Muh mein baar baar ulcer ho jata hai jo jaldi theek nahi hota.",
            "Back wala daant nikal raha hai aur bahut dard ho raha hai.",
            "Daanto mein cavity ho gayi lagti hai, khane mein fasta hai kuch.",
            "Muh se badbu aati hai hamesha, brush karne ke baad bhi.",
        ],
        "doctor_mentions": ["dentist ko dikhana hai", "daanto ka doctor chahiye",
                             "dant chikitsak se milna hai"],
    },

    "Veterinarian": {
        "keywords": ["kutte ko bukhar", "billi ne khana chhod diya", "pet ko ulti",
                     "animal ko chot"],
        "symptoms": [
            "Mere kutte ko do din se bukhar hai aur khana bhi nahi kha raha.",
            "Billi ne achanak khana peena chhod diya hai.",
            "Pet ko ulti ho rahi hai baar baar, chal bhi nahi pa raha thik se.",
            "Gaay bimaar lag rahi hai, khada nahi ho pa rahi.",
        ],
        "doctor_mentions": ["veterinarian ko dikhana hai", "pashu chikitsak chahiye",
                             "animal doctor se milna hai"],
    },
}

# Sanity check: this must match the 30 canonical categories (from
# seed_medical_specialities.sql) + Dentist + Veterinarian = 32.
CANONICAL_DB_CATEGORIES = {
    "Anaesthesiologist", "Dermatologist (Skin)", "Emergency Medicine Specialist",
    "General Physician", "Geriatrician", "Paediatrician", "Pathologist",
    "Physiotherapist / Rehab", "Psychiatrist", "Oncologist (Cancer)", "Radiologist",
    "Pulmonologist (Chest/Lungs)", "Sports Medicine Specialist", "General Surgeon",
    "Gynaecologist", "Ophthalmologist (Eye)", "Orthopaedic Surgeon (Bone)", "ENT Specialist",
    "Cardiologist (Heart)", "Rheumatologist", "Endocrinologist (Hormones/Diabetes)",
    "Gastroenterologist", "Nephrologist (Kidney)", "Neurologist",
    "GI/Surgical Gastroenterologist", "Urologist", "Vascular Surgeon",
    "Cardiothoracic Surgeon", "Neurosurgeon", "Plastic Surgeon",
}
NON_DB_CATEGORIES = {"Dentist", "Veterinarian"}

if __name__ == "__main__":
    all_expected = CANONICAL_DB_CATEGORIES | NON_DB_CATEGORIES
    seeded = set(SEED.keys())
    missing = all_expected - seeded
    extra = seeded - all_expected
    print(f"Seeded classes: {len(seeded)} / expected {len(all_expected)}")
    if missing:
        print("MISSING:", missing)
    if extra:
        print("UNEXPECTED EXTRA:", extra)
    total_phrases = sum(len(v["symptoms"]) + len(v["doctor_mentions"]) for v in SEED.values())
    print(f"Total seed phrases (symptoms + doctor_mentions): {total_phrases}")
