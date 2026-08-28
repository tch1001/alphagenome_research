# OpenSplice v3.3.2.2 controlled-stop structural archive

**Decision:** `controlled_stop_compiler_graph_mismatch`

The frozen v3.3.2 attempt stopped before any model apply because its 
fresh backend-compiled HLO was not byte-identical to the frozen v3.3 
backend artifact. StableHLO and pre-backend HLO were identical.

This archive contains **no biological evidence**: zero endpoint records, 
zero model applies, and no completed ID-0 or ID-255 controls.

**Scientific summary computed:** no  
**Shapley, interaction, resolution, or nomination computed:** no  
**Combined analysis permitted:** no

v3.3.2.2 changes only Python function-reference control flow. The saved 
frozen validator was called directly and restored exactly.

Confirmation model outputs, activations, and interventions remained 
unopened.
