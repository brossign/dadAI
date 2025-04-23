# 📦 Makefile pour automatiser les scripts de préparation des données DadAI
# A la racine: make -C scripts

DATA_DIR=./data
SCRIPTS_DIR=./scripts

.PHONY: all collect format clean

# 🧪 Pipeline complet : collecte → formatage → nettoyage
all: collect format clean

# 🔄 Étape 1 : collecte depuis Reddit
collect:
	python $(SCRIPTS_DIR)/collect_reddit_data.py

# 🧹 Étape 2 : formatage en prompt / completion
format:
	python $(SCRIPTS_DIR)/format_reddit_data.py

# 🧼 Étape 3 : nettoyage des données
clean:
	python $(SCRIPTS_DIR)/clean_dataset.py