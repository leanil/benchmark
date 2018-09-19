entry BaselineSum (A: []f32): f32 =
  reduce (+) 0 A
  
entry BaselineProd (c: f32) (A: []f32): []f32 =
  map (*c) A
  
entry Dot (A: []f32) (B: []f32): f32 =
  reduce (+) 0 (map2 (*) A B)
  
entry Dot1 (A: []f32) (B: []f32) (C: []f32): f32 =
  reduce (+) 0 (map2 (*) (map2 (*) A B) C)
  
entry Dot2 (A: []f32) (B: []f32) (C: []f32): f32 =
  reduce (+) 0 (map2 (*) (map2 (+) A B) C)
  
entry Dot3 (A: []f32) (B: []f32) (C: []f32) (D: []f32): f32 =
  reduce (+) 0 (map2 (*) (map2 (+) A B) (map2 (-) C D))
  
entry Dot4 (a: f32) (A: []f32) (b: f32) (B: []f32) (c: f32) (C: []f32) (d: f32) (D: []f32): f32 =
  reduce (+) 0 (map2 (*) (map2 (+) (map (*a) A) (map (*b) B)) (map2 (+) (map (*c) C) (map (*d) D)))
  
entry Dot5 (A: []f32) (B: []f32) (C: []f32): f32 =
  reduce (+) 0 (map2 (*) (map2 (+) A B) (map2 (-) A C))
  
entry Dot6 (A: []f32) (B: []f32) (C: []f32) (D: []f32): f32 =
  let t1 = map2 (+) A B
  let t2 = map2 (-) C D in
  reduce (+) 0 (map2 (*) t1 t2)