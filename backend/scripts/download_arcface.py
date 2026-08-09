from insightface.app import FaceAnalysis


app = FaceAnalysis(
    name="antelopev2"
)

app.prepare(
    ctx_id=-1
)

print("InsightFace antelopev2 loaded")