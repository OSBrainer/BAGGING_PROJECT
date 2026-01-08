import joblib
import pandas as pd
import numpy as np
from web3 import Web3
from solcx import compile_source, install_solc, set_solc_version
import concurrent.futures  # <--- Bibliothèque pour le Parallélisme
import time

# --- CONFIGURATION ---
MODEL_FILE = "/home/abdo/vscode/projet_bagging_blockchain/Credit-Card-Fraud-Detection-using-Random-Forest/classifier.pkl"
DATA_FILE = "/home/abdo/vscode/projet_bagging_blockchain/Credit-Card-Fraud-Detection-using-Random-Forest/creditcard.csv"
GANACHE_URL = "http://127.0.0.1:7545"

# --- SMART CONTRACT ---
contract_code = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FraudDetection {
    struct Vote {
        uint treeId;
        int prediction;
        uint timestamp;
    }
    mapping(uint => Vote[]) public sessionVotes;

    function submitVote(uint _transactionId, uint _treeId, int _prediction) public {
        sessionVotes[_transactionId].push(Vote(_treeId, _prediction, block.timestamp));
    }

    function getVerdict(uint _transactionId) public view returns (string memory, uint, uint) {
        Vote[] memory votes = sessionVotes[_transactionId];
        uint fraudes = 0;
        uint ok = 0;
        
        for(uint i=0; i < votes.length; i++){
            if(votes[i].prediction == 1) fraudes++;
            else ok++;
        }

        if (fraudes > ok) return ("FRAUDE CONFIRMEE", fraudes, ok);
        else return ("TRANSACTION VALIDE", fraudes, ok);
    }
}
'''

# --- FONCTION DU WORKER (Un Nœud Unique) ---
def node_worker(tree_id, tree_model, features, transaction_id, contract_address, account_address):
    """
    Cette fonction représente le travail d'un seul nœud dans le diagramme PCAM.
    Elle sera exécutée en parallèle sur un thread distinct.
    """
    try:
        w3_thread = Web3(Web3.HTTPProvider(GANACHE_URL))
        contract = w3_thread.eth.contract(address=contract_address, abi=CONTRACT_ABI)
        
        # 2. IA (Partition): Prédiction locale
        prediction = int(tree_model.predict(features)[0])
        
        # 3. Blockchain (Communication): Envoi de la transaction
        
        tx_hash = contract.functions.submitVote(transaction_id, tree_id, prediction).transact({
            'from': account_address,
            'gas': 200000
        })
        # w3_thread.eth.wait_for_transaction_receipt(tx_hash)

        emoji = "🔴" if prediction == 1 else "🟢"
        return f"✅ Node {tree_id:02d} [{account_address[:5]}..] a voté {emoji}"
        
    except Exception as e:
        return f"❌ Node {tree_id:02d} Erreur: {e}"

# Variable globale pour l'ABI (nécessaire pour les workers)
CONTRACT_ABI = None

def main():
    global CONTRACT_ABI
    print("--- ⚡ ARCHITECTURE PARALLÈLE (PCAM) ⚡ ---")

    # 1. Setup Blockchain
    w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
    if not w3.is_connected():
        print("❌ Ganache non connecté")
        return
    
    # Récupération de tous les comptes pour simuler des nœuds distincts
    accounts = w3.eth.accounts
    w3.eth.default_account = accounts[0]
    print(f"🔗 {len(accounts)} Comptes disponibles pour le mapping.")

    # 2. Déploiement
    print("⚙️  Déploiement du Juge (Smart Contract)...")
    install_solc('0.8.0')
    set_solc_version('0.8.0')
    compiled = compile_source(contract_code, output_values=['abi', 'bin'], solc_version='0.8.0')
    contract_id, contract_interface = compiled.popitem()
    CONTRACT_ABI = contract_interface['abi'] # Stockage global pour les threads
    
    RF = w3.eth.contract(abi=contract_interface['abi'], bytecode=contract_interface['bin'])
    tx_hash = RF.constructor().transact()
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    contract_address = receipt.contractAddress
    contract = w3.eth.contract(address=contract_address, abi=CONTRACT_ABI)

    # 3. Data & Model
    print("📂 Chargement IA...")
    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    
    # Sélection d'une fraude
    sample = df[df['Class'] == 1].sample(n=1)
    transaction_id = int(sample.index[0])
    features = sample.drop('Class', axis=1).values.reshape(1, -1)
    
    print(f"💳 Transaction à auditer : #{transaction_id} (Fraude Réelle)")

    # 4. EXÉCUTION PARALLÈLE (C'est ici que ça change)
    arbres = model.estimators_[:20] # On prend 20 arbres
    print(f"\n🚀 Lancement des {len(arbres)} nœuds en PARALLÈLE...")
    
    start_time = time.time()
    
    # Création du Pool de Threads (Comme des ouvriers virtuels)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for i, arbre in enumerate(arbres):
            # MAPPING : On assigne un compte Ethereum différent à chaque arbre (modulo 10 si on a que 10 comptes)
            node_acc = accounts[i % len(accounts)]
            
            # On soumet la tâche au pool
            futures.append(executor.submit(node_worker, i, arbre, features, transaction_id, contract_address, node_acc))
        
        # AGGLOMERATION : On récupère les résultats au fil de l'eau
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    end_time = time.time()
    print(f"\n⏱️  Temps d'exécution parallèle : {end_time - start_time:.2f} secondes")
    # 5. Verdict
    print("\n⏳ Attente du consensus blockchain...")
    time.sleep(2)
    verdict = contract.functions.getVerdict(transaction_id).call()
    print(f"\n⚖️  VERDICT FINAL : {verdict[0]} (Fraudes: {verdict[1]} vs Valide: {verdict[2]})")

if __name__ == "__main__":
    main()