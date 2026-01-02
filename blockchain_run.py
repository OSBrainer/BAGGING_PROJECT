import joblib
import numpy as np
from web3 import Web3
from solcx import compile_source, install_solc, set_solc_version
import sys
import time

# ================= CONFIGURATION =================
# Mettez ici le nom EXACT de votre fichier modèle
MODEL_FILENAME = "classifier.pkl" 

# URL de Ganache (Vérifiez si c'est 7545 ou 8545 dans l'app)
GANACHE_URL = "http://127.0.0.1:7545"
# =================================================

# --- 1. CODE DU SMART CONTRACT (SOLIDITY) ---
# Ce code sera compilé et déployé à la volée
contract_source_code = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract RandomForestDecision {
    
    struct Vote {
        uint treeId;     // Quel arbre a voté ?
        int prediction;  // 0 ou 1
        uint timestamp;  // Quand ?
    }
    
    Vote[] public votes;
    event VoteRecu(uint treeId, int prediction);

    // Fonction pour enregistrer un vote (coûte du Gas)
    function submitVote(uint _treeId, int _prediction) public {
        votes.push(Vote(_treeId, _prediction, block.timestamp));
        emit VoteRecu(_treeId, _prediction);
    }

    // Fonction pour lire le résultat final (Gratuit)
    function getConsensus() public view returns (string memory, int, int) {
        int count0 = 0;
        int count1 = 0;
        
        for(uint i=0; i<votes.length; i++){
            if(votes[i].prediction == 0) count0++;
            else count1++;
        }
        
        string memory verdict = (count1 > count0) ? "CLASSE 1" : "CLASSE 0";
        return (verdict, count0, count1);
    }
}
'''

def main():
    print(f"🚀 Démarrage du système avec le modèle : {MODEL_FILENAME}")

    # --- 2. CONNEXION BLOCKCHAIN ---
    try:
        w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
        if not w3.is_connected():
            print(f"❌ Erreur : Impossible de se connecter à Ganache sur {GANACHE_URL}")
            print("👉 Avez-vous lancé l'application Ganache ?")
            return
        print(f"✅ Connecté à Ganache. Block actuel : {w3.eth.block_number}")
    except Exception as e:
        print(f"❌ Erreur de connexion : {e}")
        return

    # Configuration du compte payeur (le premier de la liste)
    w3.eth.default_account = w3.eth.accounts[0]

    # --- 3. DEPLOIEMENT DU SMART CONTRACT ---
    print("\n⚙️  Compilation et déploiement du Smart Contract...")
    try:
        # --- CORRECTION ICI ---
        install_solc('0.8.0')     # On s'assure qu'il est là
        set_solc_version('0.8.0') # ON FORCE L'ACTIVATION DE LA VERSION <--- C'est la ligne magique
        
        compiled_sol = compile_source(
            contract_source_code,
            output_values=['abi', 'bin'],
            solc_version='0.8.0' # On précise la version ici aussi pour être sûr
        )
        # ----------------------
        
        contract_id, contract_interface = compiled_sol.popitem()
        
        bytecode = contract_interface['bin']
        abi = contract_interface['abi']

        # Déploiement
        RF_Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
        tx_hash = RF_Contract.constructor().transact()
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        contract_address = tx_receipt.contractAddress
        contract = w3.eth.contract(address=contract_address, abi=abi)
        print(f"✅ Smart Contract actif à l'adresse : {contract_address}")
    except Exception as e:
        print(f"❌ Erreur lors du déploiement du contrat : {e}")
        return

    # --- 4. CHARGEMENT DU MODELE .PKL ---
    print(f"\n📂 Chargement du fichier '{MODEL_FILENAME}'...")
    try:
        model = joblib.load(MODEL_FILENAME)
    except FileNotFoundError:
        print(f"❌ Le fichier {MODEL_FILENAME} est introuvable dans ce dossier.")
        return

    # Vérification des dimensions attendues par le modèle
    try:
        n_features = model.n_features_in_
        print(f"ℹ️  Le modèle attend {n_features} colonnes en entrée.")
    except AttributeError:
        # Si le modèle est vieux ou pas standard, on suppose une valeur par défaut
        print("⚠️ Impossible de lire n_features_in_, on essaye avec 10 features...")
        n_features = 10

    # Création d'une fausse donnée pour tester (Simulation)
    # Dans la vraie vie, ce serait les données de votre formulaire
    donnee_test = np.random.rand(1, n_features)

    # --- 5. EXECUTION DU BAGGING SUR BLOCKCHAIN ---
    print("\n🌲 --- DÉBUT DU VOTE DÉCENTRALISÉ ---")
    
    # On récupère les arbres individuels du Random Forest
    try:
        arbres = model.estimators_
    except AttributeError:
        print("❌ Erreur : Ce modèle n'est pas un Random Forest (pas d'attribut estimators_).")
        return

    print(f"ℹ️  Nombre de learners (noeuds) trouvés : {len(arbres)}")

    for i, arbre in enumerate(arbres):
        # A. Prédiction locale par l'arbre
        prediction = int(arbre.predict(donnee_test)[0])
        
        print(f"   Node {i+1}/{len(arbres)} vote : {prediction}", end=" ")
        
        # B. Envoi de la transaction (Mining)
        try:
            tx_hash = contract.functions.submitVote(i, prediction).transact()
            # On attend que le bloc soit confirmé
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"-> 🧱 Bloc miné (Gas: {receipt.gasUsed})")
        except Exception as e:
            print(f"-> ❌ Erreur Transaction : {e}")

    # --- 6. RESULTAT FINAL ---
    print("\n⚖️  --- CONSULTATION DU SMART CONTRACT ---")
    verdict, v0, v1 = contract.functions.getConsensus().call()
    
    print(f"🗳️  Total Votes '0' : {v0}")
    print(f"🗳️  Total Votes '1' : {v1}")
    print(f"🏆 RÉSULTAT FINAL CERTIFIÉ : {verdict}")

if __name__ == "__main__":
    main()