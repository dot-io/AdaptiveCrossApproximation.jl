1. Make the api more intuitive by wrapping
_assemble_rows and _assemble_cols in a nextrc!() call
2. add kwargs (default false) for the svd "compression" and
fnorm "iteration" in the batched aca impl
