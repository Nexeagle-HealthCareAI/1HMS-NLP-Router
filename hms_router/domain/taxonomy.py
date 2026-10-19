"""Specialty taxonomy: the 30 NexEagle specialtyIds and the maps onto them (one source of truth).

The ids are NexEagle's `specialtyId` slugs - the same ones Doctor Dekho, NexEagleWebsite and the
WhatsApp bot already use (see specialty_mapping.py on the `main` branch, which this replaces for
hms_router; served to callers at GET /v1/taxonomy). Two source vocabularies map onto them:

  DATASET_SPECIALTY_TO_ID  - the `specialty` column of Datasets/combined_symptom_dataset.csv
  LEGACY_LABEL_TO_ID       - the `specialist` column of Datasets/Hinglish_Symptoms_V28.csv
                             (the old nlp_brain training set)
"""
from __future__ import annotations

SPECIALTY_NAMES: dict[str, str] = {
    "general": "General Physician",
    "pediatrics": "Paediatrics",
    "cardiology": "Cardiology",
    "dermatology": "Dermatology",
    "orthopedics": "Orthopaedics",
    "gynecology": "Gynaecology",
    "dentistry": "Dentistry",
    "ent": "ENT",
    "ophthalmology": "Ophthalmology",
    "neurology": "Neurology",
    "psychiatry": "Psychiatry",
    "urology": "Urology",
    "gastroenterology": "Gastroenterology",
    "endocrinology": "Endocrinology",
    "pulmonology": "Pulmonology",
    "nephrology": "Nephrology",
    "oncology": "Oncology",
    "rheumatology": "Rheumatology",
    "physiotherapy": "Physiotherapy",
    "generalsurgery": "General Surgery",
    "neurosurgery": "Neurosurgery",
    "plasticsurgery": "Plastic Surgery",
    "vascularsurgery": "Vascular Surgery",
    "cardiothoracicsurgery": "Cardiothoracic Surgery",
    "anesthesiology": "Anaesthesiology",
    "radiology": "Radiology",
    "pathology": "Pathology",
    "emergencymedicine": "Emergency Medicine",
    "geriatrics": "Geriatrics",
    "sportsmedicine": "Sports Medicine",
}

DATASET_SPECIALTY_TO_ID: dict[str, str] = {
    "Anaesthesiology": "anesthesiology",
    "Breast Surgery": "generalsurgery",          # no breast-clinic id on the site; surgery is the closest bucket
    "Cardiology": "cardiology",
    "Cardiothoracic Surgery": "cardiothoracicsurgery",
    "Dental": "dentistry",
    "Dermatology": "dermatology",
    "ENT": "ent",
    "Emergency Medicine": "emergencymedicine",
    "Endocrinology": "endocrinology",
    "GI Surgery": "generalsurgery",              # same choice specialty_mapping.py made for GI/Surgical Gastro
    "Gastroenterology": "gastroenterology",
    "General Medicine": "general",
    "General Surgery": "generalsurgery",
    "Geriatrics": "geriatrics",
    "Gynaecology": "gynecology",
    "Nephrology": "nephrology",
    "Neurology": "neurology",
    "Neurosurgery": "neurosurgery",
    "Oncology": "oncology",
    "Ophthalmology": "ophthalmology",
    "Orthopaedics": "orthopedics",
    "Paediatrics": "pediatrics",
    "Pathology": "pathology",
    "Physiotherapy": "physiotherapy",
    "Plastic Surgery": "plasticsurgery",
    "Psychiatry": "psychiatry",
    "Pulmonology": "pulmonology",
    "Radiology": "radiology",
    "Rheumatology": "rheumatology",
    "Sports Medicine": "sportsmedicine",
    "Urology": "urology",
    "Vascular Surgery": "vascularsurgery",
}

LEGACY_LABEL_TO_ID: dict[str, str | None] = {
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
    "GI/Surgical Gastroenterologist": "generalsurgery",
    "Veterinarian": None,                        # not a human-medicine category on the site - dropped
}


def specialty_name(specialty_id: str) -> str:
    return SPECIALTY_NAMES.get(specialty_id, specialty_id)
