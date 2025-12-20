from solcx import install_solc, get_installed_solc_versions

print("🔄 Tentative d'installation manuelle de Solidity 0.8.0...")
try:
    # On force le téléchargement
    install_solc('0.8.0')
    print("✅ SUCCÈS ! Compilateur 0.8.0 installé.")
    
    # On vérifie qu'il est bien là
    print(f"versions disponibles : {get_installed_solc_versions()}")

except Exception as e:
    print(f"❌ Erreur d'installation : {e}")