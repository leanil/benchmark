# bench1: movdqa
    .globl	bench1
	.type	bench1,@function
bench1:
	.cfi_startproc
	addq %rdi, %rsi
1:	movq %rdi, %rax
	.p2align	4, 0x90
2:	movdqa 0*16(%rax), %xmm0
	movdqa 1*16(%rax), %xmm0
	movdqa 2*16(%rax), %xmm0
	movdqa 3*16(%rax), %xmm0
	movdqa 4*16(%rax), %xmm0
	movdqa 5*16(%rax), %xmm0
	movdqa 6*16(%rax), %xmm0
	movdqa 7*16(%rax), %xmm0
	movdqa 8*16(%rax), %xmm0
	movdqa 9*16(%rax), %xmm0
	movdqa 10*16(%rax), %xmm0
	movdqa 11*16(%rax), %xmm0
	movdqa 12*16(%rax), %xmm0
	movdqa 13*16(%rax), %xmm0
	movdqa 14*16(%rax), %xmm0
	movdqa 15*16(%rax), %xmm0
	addq    $16*16, %rax
	cmpq    %rsi, %rax
	jb     2b
	dec    %rdx
    jnz    1b
	ret
	.cfi_endproc

# bench2: vmovdqa
	.globl	bench2
	.type	bench2,@function
bench2:
	.cfi_startproc
	addq %rdi, %rsi
1:	movq %rdi, %rax
	.p2align	4, 0x90
2:	vmovdqa 0*32(%rax), %ymm0
	vmovdqa 1*32(%rax), %ymm0
	vmovdqa 2*32(%rax), %ymm0
	vmovdqa 3*32(%rax), %ymm0
	vmovdqa 4*32(%rax), %ymm0
	vmovdqa 5*32(%rax), %ymm0
	vmovdqa 6*32(%rax), %ymm0
	vmovdqa 7*32(%rax), %ymm0
	addq    $8*32, %rax
	cmpq    %rsi, %rax
	jb     2b
	dec    %rdx
    jnz    1b
	ret
	.cfi_endproc

# bench3: movapd
    .globl	bench3
	.type	bench3,@function
bench3:
	.cfi_startproc
	addq %rdi, %rsi
1:	movq %rdi, %rax
	.p2align	4, 0x90
2:	movapd 0*16(%rax), %xmm0
	movapd 1*16(%rax), %xmm0
	movapd 2*16(%rax), %xmm0
	movapd 3*16(%rax), %xmm0
	movapd 4*16(%rax), %xmm0
	movapd 5*16(%rax), %xmm0
	movapd 6*16(%rax), %xmm0
	movapd 7*16(%rax), %xmm0
	movapd 8*16(%rax), %xmm0
	movapd 9*16(%rax), %xmm0
	movapd 10*16(%rax), %xmm0
	movapd 11*16(%rax), %xmm0
	movapd 12*16(%rax), %xmm0
	movapd 13*16(%rax), %xmm0
	movapd 14*16(%rax), %xmm0
	movapd 15*16(%rax), %xmm0
	addq    $16*16, %rax
	cmpq    %rsi, %rax
	jb     2b
	dec    %rdx
    jnz    1b
	ret
	.cfi_endproc

# bench4: vmovapd
	.globl	bench4
	.type	bench4,@function
bench4:
	.cfi_startproc
	addq %rdi, %rsi
1:	movq %rdi, %rax
	.p2align	4, 0x90
2:	vmovapd 0*32(%rax), %ymm0
	vmovapd 1*32(%rax), %ymm0
	vmovapd 2*32(%rax), %ymm0
	vmovapd 3*32(%rax), %ymm0
	vmovapd 4*32(%rax), %ymm0
	vmovapd 5*32(%rax), %ymm0
	vmovapd 6*32(%rax), %ymm0
	vmovapd 7*32(%rax), %ymm0
	addq    $8*32, %rax
	cmpq    %rsi, %rax
	jb     2b
	dec    %rdx
    jnz    1b
	ret
	.cfi_endproc
