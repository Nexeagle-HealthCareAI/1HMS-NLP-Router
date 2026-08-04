"""Voice-to-Text layer: microphone capture, speech-to-text, and Devanagari
transliteration, followed by an HTTP call into the FastAPI layer.

No import of nlp_brain belongs here -- this layer only ever talks to the NLP
Brain through the FastAPI HTTP contract (see api_client.SymptomRouterClient),
so it can be developed, tested, and deployed (e.g. on a device with a mic but
no ML dependencies) independently of the other two layers.
"""
