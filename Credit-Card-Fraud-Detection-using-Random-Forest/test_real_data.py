import joblib
import pandas as pd
import numpy as np
from web3 import Web3
from solcx import compile_source, install_solc, set_solc_version

# --- CONFIGURATION ---
MODEL_FILE = "classifier.pkl"
DATA_FILE = "creditcard.csv"
GANACHE_URL = "http://127.0.0.1:7545"

# --- SMART CONTRACT (Emojis retirés pour éviter l'erreur) ---
contract_code = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FraudDetection {
    struct Vote {
        uint treeId;
        int prediction; // 0=OK, 1=FRAUDE
        uint timestamp;
    }
    Vote[] public votes;
    
    function submitVote(uint _treeId, int _prediction) public {
        votes.push(Vote(_treeId, _prediction, block.timestamp));
    }

    function getVerdict() public view returns (string memory) {
        int fraudes = 0;
        for(uint i=0; i<votes.length; i++){
            if(votes[i].prediction == 1) fraudes++;
        }
        // Si au moins un arbre voit une fraude
        if (fraudes > 0) return "FRAUDE DETECTEE"; 
        else return "Transaction Valide";
    }
}
'''

def main():
    print("--- 🕵️ TEST SUR DONNEES REELLES (DATASET) ---")
    
    # 1. Connexion Blockchain
    w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
    if not w3.is_connected():
        print("❌ Erreur : Lancez Ganache !")
        return
    w3.eth.default_account = w3.eth.accounts[0]

    # 2. Déploiement Contrat
    print("⚙️  Déploiement du Smart Contract...")
    try:
        install_solc('0.8.0')
        set_solc_version('0.8.0')
        compiled = compile_source(contract_code, output_values=['abi', 'bin'], solc_version='0.8.0')
        contract_id, contract_interface = compiled.popitem()
        
        RF = w3.eth.contract(abi=contract_interface['abi'], bytecode=contract_interface['bin'])
        tx_hash = RF.constructor().transact()
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        contract = w3.eth.contract(address=receipt.contractAddress, abi=contract_interface['abi'])
        print(f"✅ Contrat prêt : {receipt.contractAddress}")
    except Exception as e:
        print(f"❌ Erreur de compilation : {e}")
        return

    # 3. Chargement des Données
    print(f"\n📂 Lecture de {DATA_FILE} (Cela peut prendre quelques secondes)...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print("❌ Fichier creditcard.csv introuvable.")
        return

    # 4. CHOIX DU TEST
    print("\nQue voulez-vous tester ?")
    print("1. Une transaction NORMALE (Classe 0)")
    print("2. Une vraie FRAUDE (Classe 1)")
    choix = input("Votre choix (1 ou 2) : ")

    if choix == "2":
        # On filtre toutes les fraudes
        fraudes = df[df['Class'] == 1]
        # On en pioche UNE au hasard (.sample)
        row_df = fraudes.sample(n=1)
        row = row_df.iloc[0]
        # On affiche l'ID pour prouver que ça change
        print(f"\n⚠️  Chargement de la Fraude ID n°{row_df.index[0]}...")
    else:
        # On filtre toutes les transactions normales
        normales = df[df['Class'] == 0]
        # On en pioche UNE au hasard
        row_df = normales.sample(n=1)
        row = row_df.iloc[0]
        print(f"\n✅ Chargement de la Transaction Normale ID n°{row_df.index[0]}...")

    # Préparation des features (On enlève la colonne 'Class')
    features = row.drop('Class').values.reshape(1, -1)
    
    # 5. Chargement Modèle
    try:
        model = joblib.load(MODEL_FILE)
        arbres = model.estimators_
        print(f"🌲 Le Random Forest a {len(arbres)} arbres.")
    except FileNotFoundError:
        print(f"❌ Modèle {MODEL_FILE} introuvable.")
        return

    # 6. Exécution Blockchain
    print("\n🚀 Lancement de l'audit décentralisé...")
    fraudes_count = 0 # Variable corrigée
    
    for i, arbre in enumerate(arbres):
        prediction = int(arbre.predict(features)[0])
        
        status = "🔴 FRAUDE" if prediction == 1 else "🟢 OK"
        if prediction == 1: fraudes_count += 1
        
        print(f"   Node {i+1} vote : {status} ->", end=" ")
        
        # Envoi transaction
        try:
            tx = contract.functions.submitVote(i, prediction).transact()
            w3.eth.wait_for_transaction_receipt(tx)
            print("Bloc miné 🧱")
        except Exception as e:
            print(f"Erreur Tx: {e}")

    # 7. Verdict Final
    print("\n⚖️  VERDICT DU SMART CONTRACT :")
    verdict = contract.functions.getVerdict().call()
    
    # On rajoute les emojis ici en Python pour l'affichage final, c'est plus sûr
    if "FRAUDE" in verdict:
        print(f"⚠️  {verdict} ⚠️")
    else:
        print(f"✅ {verdict}")

if __name__ == "__main__":
    main()