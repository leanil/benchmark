let dotprod [n] (xs: [n]f32) (ys: [n]f32): f32 =
    reduce (+) 0 (map2 (*) xs ys)

let scl2d [n][m] (a: f32) (A: [n][m]f32): [n][m]f32 =
    map (\rA -> map (*a) rA) A
    
let scl3d [n][m][o] (a: f32) (A: [n][m][o]f32) : [n][m][o]f32 =
    map (scl2d a) A
    
let sum2d [n][m] (A: [n][m]f32) (B: [n][m]f32): [n][m]f32 =
    map2 (\rA rB -> map2 (+) rA rB) A B
    
let sum3d [n][m][o] (A: [n][m][o]f32) (B: [n][m][o]f32): [n][m][o]f32 =
    map2 sum2d A B
    
entry t1 [i][j][k][l] (A: [i][j][k]f32) (B: [k][l]f32): [i][j][l]f32 =
    map (\sA -> map (\rA -> map (\rB -> dotprod rA rB) (transpose B)) sA) A
    
entry t2 [i][j][k][l] (A: [i][j][k]f32) (B: [k]f32) (C: [l]f32): [i][j][l]f32 =
    let outer = map (\x -> map (\y -> x*y) C) B in
    map (\sA -> map (\rA -> map (\rB -> dotprod rA rB) (transpose outer)) sA) A
    
entry t3 [i][j][k][l] (A: [i][j][k]f32) (B: [k][l]f32) (C: [k]f32) (D: [j]f32): [i][j][l]f32 =
    map (\sA -> map2 (\rA d -> map (\rB -> reduce (+) 0 (map3 (\x y z -> x*y*z*d) rA rB C)) (transpose B)) sA D) A
    
entry t4 [i][j][k][l] (A: [i][j][k]f32) (B: [i][j][k]f32) (C: [k][l]f32): [i][j][l]f32 =
    map (\sA -> map (\rA -> map (\rC -> dotprod rA rC) (transpose C)) sA) (sum3d A B)
    
entry t5 [i][j][k][l] (A: [i][j][k]f32) (B: [i][j][k]f32) (C: [k][l]f32) (D: [k][l]f32): [i][j][l]f32 =
    map (\sA -> map (\rA -> map (\rC -> dotprod rA rC) (transpose (sum2d C D))) sA) (sum3d A B)
    
entry t6 [i][j][k][l] (a: f32) (A: [i][j][k]f32) (b: f32) (B: [i][j][k]f32) (c: f32) (C: [k][l]f32) (d: f32) (D: [k][l]f32): [i][j][l]f32 =
    let sum1 = sum3d (scl3d a A) (scl3d b B)
    let sum2 = sum2d (scl2d c C) (scl2d d D) in
    map (\sA -> map (\rA -> map (\rC -> dotprod rA rC) (transpose sum2)) sA) sum1
    
entry t7 [i][j][k][l] (a: f32) (A: [i][j][k]f32) (b: f32) (B: [i][j][k]f32) (c: f32) (C: [k][l]f32) (d: f32) (D: [k][l]f32) (e: f32) (E: [k]f32) (f: f32) (F: [k]f32) (g: f32) (G: [j]f32) (h: f32) (H: [j]f32): [i][j][l]f32 =
    let sum1 = sum3d (scl3d a A) (scl3d b B)
    let sum2 = sum2d (scl2d c C) (scl2d d D) 
    let sum3 = map2 (+) (map (*e) E) (map (*f) F)
    let sum4 = map2 (+) (map (*g) G) (map (*h) H) in
    map (\sA -> map2 (\rA d -> map (\rB -> reduce (+) 0 (map3 (\x y z -> x*y*z*d) rA rB sum3)) (transpose sum2)) sA sum4) sum1
    
entry t8 [i][j][k][l][m] (A: [i][j][k]f32) (B: [k][l]f32) (C: [j][m]f32): [i][l][m]f32 =
    map (\sA -> map (\rB -> map (\rC -> 
        reduce (+) 0 (map2 (\rA c -> (reduce (+) 0 (map2 (*) rA rB))*c) sA rC)
        ) (transpose C)) (transpose B)) A
        
entry t9 [i][j][k][l] (A: [i][j][k]f32) (B: [k][l]f32) (C: [l]f32): [i][j]f32 =
    let sum = map (\rB -> reduce (+) 0 (map2 (*) rB C)) B in
    map (\sA -> map (\rA -> reduce (+) 0 (map2 (*) rA sum)) sA) A
    
entry t10 [i][j][k][l] (A: [i][j][k]f32) (B: [i][k][l]f32) (C: [l]f32): [i][j]f32 =
    let sum = map (\sB -> map (\rB -> reduce (+) 0 (map2 (*) rB C)) sB) B in
    map2 (\sA rB -> map (\rA -> reduce (+) 0 (map2 (*) rA rB)) sA) A sum
    
entry t11 [i][j][k][l] (A: [i]f32) (B: [j]f32) (C: [k]f32) (D: [i][k][l]f32) (E: [l]f32): [i][j]f32 =
    let sum1 = map (\a -> map (\b -> map (\c -> a*b*c) C) B) A
    let sum2 = map (\sD -> map (\rD -> reduce (+) 0 (map2 (*) rD E)) sD) D in
    map2 (\sA rB -> map (\rA -> reduce (+) 0 (map2 (*) rA rB)) sA) sum1 sum2