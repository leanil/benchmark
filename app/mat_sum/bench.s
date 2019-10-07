    .globl	mat_sum_5
	.type	mat_sum_5,@function
mat_sum_5:
	.cfi_startproc
	pushq	%r12
	shlq	$3, %rdx
	lea		(%rdi, %rdx), %r9
	movq	%rdx, %r10
	imulq	%rsi, %r10
	imulq	$3, %rdx, %r11
	imulq	$5, %rdx, %r12
1:	movq	%rdi, %rax
	xorpd	%xmm0, %xmm0
	xorpd	%xmm1, %xmm1
	xorpd	%xmm2, %xmm2
	xorpd	%xmm3, %xmm3
	xorpd	%xmm4, %xmm4
	xorpd	%xmm5, %xmm5
2: 	lea		(%rax, %r10), %r8
	.p2align	4, 0x90
3:	addpd	(%rax),			 %xmm0
	addpd	(%rax, %rdx),	 %xmm1
	addpd	(%rax, %rdx, 2), %xmm2
	addpd	(%rax, %r11), 	 %xmm3
	addpd	(%rax, %rdx, 4), %xmm4
	addpd	(%rax, %r12), 	 %xmm5
	lea		(%rax, %r11, 2), %rax
	cmpq	%r8, %rax
	jb		3b
	subq	%r10, %rax
	addq	$16, %rax
	cmpq	%r9, %rax
	jb		2b
	addpd	%xmm5, %xmm1
	addpd	%xmm4, %xmm1
	addpd	%xmm3, %xmm1
	addpd	%xmm2, %xmm1
	addpd	%xmm0, %xmm1
	movapd	%xmm1, %xmm0
	unpckhpd	%xmm1, %xmm0
	addpd	%xmm1, %xmm0
	dec		%rcx
    jnz		1b
	popq	%r12
	ret
	.cfi_endproc
