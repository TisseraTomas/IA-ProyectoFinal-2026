class IntentGuardrail:
    def __init__(self, vector_db, distance_threshold: float = 1.2):
        self.vector_db = vector_db
        # Umbral de distancia: en Chroma con L2, distancias menores a 1.2 o 1.4 
        # indican alta cercanía semántica.
        self.threshold = distance_threshold

    def evaluate_query(self, user_query: str):
        # Usamos similarity_search_with_score que devuelve distancia (cuanto más chico, más cercano)
        results = self.vector_db.similarity_search_with_score(user_query, k=1)
        
        if not results:
            return {"allowed": False, "reason": "No match", "score": 0.0, "data": None}

        doc, distance = results[0]

        # Si la distancia es mayor al umbral, significa que está muy lejos semánticamente (Out of Domain)
        if distance > self.threshold:
            return {
                "allowed": False,
                "reason": "Out of Domain / Potential Jailbreak",
                "score": distance,
                "data": None
            }

        return {
            "allowed": True,
            "reason": "In Domain",
            "score": distance,
            "data": doc.metadata
        }