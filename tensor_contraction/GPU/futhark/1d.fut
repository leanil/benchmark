entry dot [n] (A: [n]f32) (B: [n]f32): f32 =
    reduce (+) 0 (map2 (*) A B)

entry dot1 [n] (A: [n]f32) (B: [n]f32) (C: [n]f32): f32 =
    reduce (+) 0 (map2 (*) (map2 (*) A B) C)

entry dot2 [n] (A: [n]f32) (B: [n]f32) (C: [n]f32): f32 =
    reduce (+) 0 (map2 (*) (map2 (+) A B) C)

entry dot3 [n] (A: [n]f32) (B: [n]f32) (C: [n]f32): f32 =
    reduce (+) 0 (map2 (*) (map2 (+) A B) (map2 (-) A C))

entry dot4 [n](A: [n]f32) (B: [n]f32) (C: [n]f32) (D: [n]f32): f32 =
    reduce (+) 0 (map2 (*) (map2 (+) A B) (map2 (-) C D))

entry dot5 [n] (a: f32) (A: [n]f32) (b: f32) (B: [n]f32) (c: f32) (C: [n]f32) (d: f32) (D: [n]f32): f32 =
    reduce (+) 0 (map2 (*) (map2 (+) (map (*a) A) (map (*b) B)) (map2 (+) (map (*c) C) (map (*d) D)))

entry dot6 (A: []f32) (B: []f32) (C: []f32) (D: []f32): f32 =
    let t1 = map2 (+) A B
    let t2 = map2 (-) C D in
    reduce (+) 0 (map2 (*) t1 t2)