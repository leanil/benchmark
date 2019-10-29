    .globl	array_sum_asm_nomov
	.type	array_sum_asm_nomov,@function
array_sum_asm_nomov:
	.cfi_startproc
	shlq	$3, %rsi
	addq	%rdi, %rsi
	xorpd	%xmm0, %xmm0
	xorpd	%xmm1, %xmm1
	xorpd	%xmm2, %xmm2
	xorpd	%xmm3, %xmm3
	xorpd	%xmm4, %xmm4
	xorpd	%xmm5, %xmm5
	.p2align	4, 0x90
1:	addpd	0*16(%rdi), %xmm0
	addpd	1*16(%rdi), %xmm1
	addpd	2*16(%rdi), %xmm2
	addpd	3*16(%rdi), %xmm3
	addpd	4*16(%rdi), %xmm4
	addpd	5*16(%rdi), %xmm5
	addq	$6*16, %rdi
	cmpq	%rsi, %rdi
	jb	1b
	addpd	%xmm1, %xmm0
	addpd	%xmm2, %xmm0
	addpd	%xmm3, %xmm0
	addpd	%xmm4, %xmm0
	addpd	%xmm5, %xmm0
	movapd	%xmm0, %xmm1
	unpckhpd	%xmm0, %xmm1
	addpd	%xmm1, %xmm0
	ret
	.cfi_endproc

    .globl	array_sum_asm_mov
	.type	array_sum_asm_mov,@function
array_sum_asm_mov:
	.cfi_startproc
	shlq	$3, %rsi
	addq	%rdi, %rsi
	xorpd	%xmm0, %xmm0
	xorpd	%xmm1, %xmm1
	xorpd	%xmm2, %xmm2
	xorpd	%xmm3, %xmm3
	xorpd	%xmm4, %xmm4
	xorpd	%xmm5, %xmm5
	.p2align	4, 0x90
1:
	movapd	0*16(%rdi), %xmm6
	addpd	%xmm6, %xmm0
	movapd	1*16(%rdi), %xmm7
	addpd	%xmm7, %xmm1
	movapd	2*16(%rdi), %xmm8
	addpd	%xmm8, %xmm2
	movapd	3*16(%rdi), %xmm9
	addpd	%xmm9, %xmm3
	movapd	4*16(%rdi), %xmm10
	addpd	%xmm10, %xmm4
	movapd	5*16(%rdi), %xmm11
	addpd	%xmm11, %xmm5
	addq	$6*16, %rdi
	cmpq	%rsi, %rdi
	jb	1b
	addpd	%xmm1, %xmm0
	addpd	%xmm2, %xmm0
	addpd	%xmm3, %xmm0
	addpd	%xmm4, %xmm0
	addpd	%xmm5, %xmm0
	movapd	%xmm0, %xmm1
	unpckhpd	%xmm0, %xmm1
	addpd	%xmm1, %xmm0
	ret
	.cfi_endproc

