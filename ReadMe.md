
# Get max sdk

### On ARM64 Install libomp (VIA BREW ONLY)

    brew install libomp

### Clone the Max SDK repository
   
     chmod +x clone_maxsdk.sh
    ./clone_maxsdk.sh

### Post Build Verification

    chmod +x verification.sh
    

    # Verify all .mxo bundles in the directory
    ./verification.sh
    
    # Verify a specific bundle
    ./verification.sh frft.mxo
    
    # Enable Gatekeeper check (for distribution on macOS)
    CHECK_GATEKEEPER=1 ./verification.sh