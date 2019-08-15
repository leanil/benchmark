entry t1_ [i][j] (A: [i][j]f32) (B: [i]f32): [j]f32 = 
    reduce (\r1 r2 -> map2 (+) r1 r2)
           (replicate j 0)
           (map2 (\rA b -> map (*b) rA) A B)
           
entry t1 [i][j] (A: [i][j]f32) (B: [j]f32): [i]f32 = 
    map (\rA -> reduce (+) 0 (map2 (*) rA B)) A
           
entry t2 [i][j] (A: [i][j]f32) (B: [j]f32) (C: [i]f32): [i]f32 =
    map2 (\rA c -> reduce (+) 0 (map2 (\a b -> a*b*c) rA B)) A C
    -- map2 (*) (t1 A B) C
    
entry t3 [i][j] (A: [i][j]f32) (B: [i][j]f32) (C: [j]f32): [i]f32 =
    map2 (\rA rB -> reduce (+) 0 (map2 (*) (map2 (+) rA rB) C)) A B
    
entry t4 [i][j] (A: [i][j]f32) (B: [i][j]f32) (C: [j]f32) (D: [j]f32): [i]f32 =
    map2 (\rA rB -> reduce (+) 0 
                           (map2 (*) 
                                 (map2 (+) rA rB)
                                 (map2 (+) C D)))
         A B
                                 
entry t5 [i][j] (a: f32) (A: [i][j]f32) (b: f32) (B: [i][j]f32) (c: f32) (C: [j]f32) (d: f32) (D: [j]f32): [i]f32 =
    map2 (\rA rB -> reduce (+) 0 
                           (map2 (*) 
                                 (map2 (+) (map (*a) rA) (map (*b) rB))
                                 (map2 (+) (map (*c) C) (map (*d) D))))
         A B
         
entry t6 [i][j] (A: [i]f32) (B: [j]f32) (C: [i]f32) (D: [j]f32): [i]f32 =
    map2 (\a c -> reduce (+) 0 (map2 (\b d -> a*b*c*d) B D)) A C
    
entry t7 [i][j][k] (A: [i][j]f32) (C: [j][k]f32) (D: [k]f32): [i]f32 =
    map (\rA -> reduce (+) 0 (map2 (*) rA 
                                       (map (\rC -> reduce (+) 0 (map2 (*) rC D)) C)))
        A
        
entry t8 [i][j][k] (A: [i][j]f32) (B: [i][j]f32) (C: [j][k]f32) (D: [k]f32): [i]f32 =
    map2 (\rA rB -> reduce (+) 0 (map2 (*) (map2 (+) rA rB) 
                                           (map (\rC -> reduce (+) 0 (map2 (*) rC D)) C)))
        A B
        
entry t9 [i][j][k] (A: [i][k]f32) (B: [k][j]f32) (C: [j]f32) (D: [j]f32): [i]f32 =
    map (\rProd -> reduce (+) 0 (map2 (*) rProd (map2 (+) C D)))
        (map (\rA -> map (\cB -> reduce (+) 0 (map2 (*) rA cB)) (transpose B)) A)
        
entry t10 [i][j][k] (A: [i][k]f32) (B: [k][j]f32) (C: [j][k]f32) (D: [k]f32): [i]f32 =
    map (\rProd -> reduce (+) 0 (map2 (*) rProd 
                                          (map (\rC -> reduce (+) 0 (map2 (*) rC D)) C)))
        (map (\rA -> map (\cB -> reduce (+) 0 (map2 (*) rA cB)) (transpose B)) A)