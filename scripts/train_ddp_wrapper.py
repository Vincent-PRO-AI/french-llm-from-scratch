#!/usr/bin/env python3
"""
DDP Wrapper - Lance le training en mode DDP
Auto-gère --local-rank depuis torch.distributed.launch
"""
import os
import sys

# Récupérer le local_rank depuis os.environ (mis par torch.distributed.launch)
local_rank = int(os.environ.get("LOCAL_RANK", 0))

# Nettoyer sys.argv du --local-rank
new_argv = []
skip_next = False
for i, arg in enumerate(sys.argv):
    if skip_next:
        skip_next = False
        continue
    if arg.startswith("--local-rank"):
        if "=" not in arg:  # format: --local-rank 0
            skip_next = True
        # Skip --local-rank=X ou --local-rank X
        continue
    new_argv.append(arg)

sys.argv = new_argv

# Maintenant importer et lancer
from train_subtitles_transformer import main

if __name__ == "__main__":
    main()

