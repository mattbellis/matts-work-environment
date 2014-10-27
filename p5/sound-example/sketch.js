var t = 0.0;
var osc0 = new p5.SinOsc(440);
var osc1 = new p5.SinOsc(660);

function setup() {
  // put setup code here
    createCanvas(900, 600);    // **change** size() to createCanvas()
    stroke(255);               // stroke() is the same
    osc0.amp(0.5);
    osc1.amp(0.1);
    //noFill();                  // noFill() is the same
}

function draw() {
  // put drawing code here
    background(200);
    fill(255,0,0);
    ellipse(50,50,20*sin(t/10.0),40);

    if (t<1)
    {
    osc0.start();
    osc1.start();
    }
    //osc0.freq(440+t*2);
    //osc1.freq(440-t*2);
    osc0.freq(440);
    osc1.freq(660);

    if (t>100)
    {
        osc0.stop();
        osc1.stop();
    }

    //println(t);

    t += 1.0;
}
