var t = 0.0;
var osc0 = new p5.SinOsc(440);
var osc1 = new p5.SinOsc(240);
var oscs = [new p5.SinOsc(), new p5.SinOsc()];

function setup() {
  // put setup code here
    createCanvas(900, 600);    // **change** size() to createCanvas()
    stroke(255);               // stroke() is the same
    osc0.amp(0.5);
    osc1.amp(0.5);
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
    osc0.freq(440+t);
    osc0.pan(-1+t*0.02);
    osc0.amp(0.5-0.005*t);
    osc1.freq(240+t);
    osc1.pan(1.00-t*0.02);
    osc1.amp(0.55-0.015*t);

    if (t>100)
    {
        osc0.stop();
        osc1.stop();
    }

    //println(t);

    t += 1.0;
}
