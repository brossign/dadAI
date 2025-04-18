# 🧠 On importe le wrapper de LangChain pour appeler une API type OpenAI
from langchain.chat_models import ChatOpenAI

# 📩 On importe le format du message "humain", comme dans ChatGPT (role=user)
from langchain.schema import HumanMessage

# 🔌 Connexion au serveur LocalAI qui tourne en local sur le port 8080
llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",  # ton API locale (LocalAI)
    api_key="not-needed",                 # LocalAI ne vérifie pas la clé
    model="mistral"                       # nom du modèle tel que défini dans config.yaml
)

# 🧾 Création du message à envoyer au modèle (comme un prompt dans ChatGPT)
messages = [
    HumanMessage(content="Je suis un nouveau papa. Comment puis-je aider ma femme qui vient d’accoucher ?")
]

# 🤖 Envoi du prompt au modèle Mistral via LangChain → appel API en local
response = llm(messages)

# 🖨️ Affichage de la réponse du modèle
print("Réponse de DadAI (Mistral) :\n")
print(response.content)