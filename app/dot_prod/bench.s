    .globl	bench6
	.type	bench6,@function
bench6:
	.cfi_startproc
	shlq	$3, %rdx
1:	xorq	%rax, %rax
	xorpd	%xmm0, %xmm0
	xorpd	%xmm1, %xmm1
	xorpd	%xmm2, %xmm2
	xorpd	%xmm3, %xmm3
	xorpd	%xmm4, %xmm4
	xorpd	%xmm5, %xmm5
	.p2align	4, 0x90
2:	
	movapd	0*16(%rdi,%rax), %xmm6
	mulpd	0*16(%rsi,%rax), %xmm6
	addpd	%xmm6, %xmm0
	movapd	1*16(%rdi,%rax), %xmm7
	mulpd	1*16(%rsi,%rax), %xmm7
	addpd	%xmm7, %xmm1
	movapd	2*16(%rdi,%rax), %xmm8
	mulpd	2*16(%rsi,%rax), %xmm8
	addpd	%xmm8, %xmm2
	movapd	3*16(%rdi,%rax), %xmm9
	mulpd	3*16(%rsi,%rax), %xmm9
	addpd	%xmm9, %xmm3
	movapd	4*16(%rdi,%rax), %xmm10
	mulpd	4*16(%rsi,%rax), %xmm10
	addpd	%xmm10, %xmm4
	movapd	5*16(%rdi,%rax), %xmm11
	mulpd	5*16(%rsi,%rax), %xmm11
	addpd	%xmm11, %xmm5
	addq	$6*16, %rax
	cmpq	%rdx, %rax
	jb	2b
	movapd	%xmm5, %xmm6
	unpckhpd	%xmm5, %xmm6    # xmm6 = xmm6[1],xmm5[1]
	addsd	%xmm5, %xmm6
	movapd	%xmm4, %xmm5
	unpckhpd	%xmm4, %xmm5    # xmm5 = xmm5[1],xmm4[1]
	addsd	%xmm6, %xmm5
	addsd	%xmm4, %xmm5
	movapd	%xmm3, %xmm4
	unpckhpd	%xmm3, %xmm4    # xmm4 = xmm4[1],xmm3[1]
	addsd	%xmm5, %xmm4
	addsd	%xmm3, %xmm4
	movapd	%xmm2, %xmm3
	unpckhpd	%xmm2, %xmm3    # xmm3 = xmm3[1],xmm2[1]
	addsd	%xmm4, %xmm3
	addsd	%xmm2, %xmm3
	movapd	%xmm0, %xmm2
	unpckhpd	%xmm0, %xmm2    # xmm2 = xmm2[1],xmm0[1]
	addsd	%xmm3, %xmm2
	addsd	%xmm0, %xmm2
	movapd	%xmm1, %xmm0
	unpckhpd	%xmm1, %xmm0    # xmm0 = xmm0[1],xmm1[1]
	addsd	%xmm2, %xmm0
	addsd	%xmm1, %xmm0
	dec    %rcx
    jnz    1b
	ret
	.cfi_endproc
