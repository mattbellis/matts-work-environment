instr 1
a1        oscil     p4, p5, 1      ; p4=amp
out       a1                       ; p5=freq
endin

instr 2
ifunc     =         p11                                ; select the basic waveform
a1        oscil     p4, p5, 1      ; p4=amp
out       a1                       ; p5=freq
endin
