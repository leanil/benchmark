entry sum [n] (A: [n]f32): f32 =
    reduce (+) 0 A
    
entry inc [n] (A: [n]f32): [n]f32 =
    map (+1) A

entry prod [n] (c: f32, A: [n]f32): [n]f32 =
    map (*c) A