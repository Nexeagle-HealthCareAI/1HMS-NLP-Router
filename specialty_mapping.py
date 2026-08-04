# -*- coding: utf-8 -*-
"""
Shared specialty-label <-> NexEagleWebsite specialtyId mapping. Split out of app.py so the
retrain pipeline (which needs the REVERSE direction, to turn feedback's specialtyId slugs
back into our internal training labels) can import it without pulling in FastAPI.
"""

# Maps our internal 32-class taxonomy to NexEagleWebsite's own `specialtyId` slugs
# (src/data/patient.ts `specialties` array). NexEagleWebsite already distinguishes the
# medical-vs-surgical siblings (neurology/neurosurgery, cardiology/cardiothoracicsurgery)
# that the original prototype's MODEL_OUTPUT_MERGES collapsed for single-label accuracy —
# nlp_brain trains WITHOUT that collapse and lets nlp_brain/candidates.py's build_candidates()
# surface both sibling specialties when genuinely ambiguous, instead of losing the distinction.
LABEL_TO_NEXEAGLE_SPECIALTY_ID = {
    "General Physician": "general",
    "Paediatrician": "pediatrics",
    "Cardiologist (Heart)": "cardiology",
    "Dermatologist (Skin)": "dermatology",
    "Orthopaedic Surgeon (Bone)": "orthopedics",
    "Gynaecologist": "gynecology",
    "Dentist": "dentistry",
    "ENT Specialist": "ent",
    "Ophthalmologist (Eye)": "ophthalmology",
    "Neurologist": "neurology",
    "Psychiatrist": "psychiatry",
    "Urologist": "urology",
    "Gastroenterologist": "gastroenterology",
    "Endocrinologist (Hormones/Diabetes)": "endocrinology",
    "Pulmonologist (Chest/Lungs)": "pulmonology",
    "Nephrologist (Kidney)": "nephrology",
    "Oncologist (Cancer)": "oncology",
    "Rheumatologist": "rheumatology",
    "Physiotherapist / Rehab": "physiotherapy",
    "General Surgeon": "generalsurgery",
    "Neurosurgeon": "neurosurgery",
    "Plastic Surgeon": "plasticsurgery",
    "Vascular Surgeon": "vascularsurgery",
    "Cardiothoracic Surgeon": "cardiothoracicsurgery",
    "Anaesthesiologist": "anesthesiology",
    "Radiologist": "radiology",
    "Pathologist": "pathology",
    "Emergency Medicine Specialist": "emergencymedicine",
    "Geriatrician": "geriatrics",
    "Sports Medicine Specialist": "sportsmedicine",
    # No distinct "surgical GI" id on NexEagleWebsite — its own "generalsurgery" blurb
    # ("Hernia, gallbladder & general operations") is the closest real bucket.
    "GI/Surgical Gastroenterologist": "generalsurgery",
    # Not a human-medicine category on a doctor-booking site — no target, dropped.
    "Veterinarian": None,
}

# Reverse direction, for turning a feedback row's specialtyId (what was predicted, or what
# was actually booked) back into one of our internal training labels. Not a clean inverse —
# "generalsurgery" is the target of two internal labels (General Surgeon and GI/Surgical
# Gastroenterologist) — General Surgeon wins as the canonical choice here since it's the
# far more common real-world case; this is a known, accepted simplification, not a bug.
NEXEAGLE_SPECIALTY_ID_TO_LABEL = {
    slug: label for label, slug in LABEL_TO_NEXEAGLE_SPECIALTY_ID.items() if slug is not None
}
# Explicit override for the documented many-to-one collision above (dict comprehension order
# would otherwise depend on LABEL_TO_NEXEAGLE_SPECIALTY_ID's insertion order, which is fragile
# to rely on implicitly).
NEXEAGLE_SPECIALTY_ID_TO_LABEL["generalsurgery"] = "General Surgeon"
