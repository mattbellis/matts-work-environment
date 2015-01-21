var t = 0.0;
var osc0 = new p5.SinOsc(440);


function setup() {
  // put setup code here
    createCanvas(900, 600);    // **change** size() to createCanvas()
    stroke(255);               // stroke() is the same
    osc0.amp(0.1);
    //noFill();                  // noFill() is the same
}

function draw() {
  // put drawing code here

    if (t<1)
    {
        osc0.start();
    }

    if (t%10==0)
    {
    var x = random(0,900);
    var y = random(0,600);
    var radius = random(0,100);

    background(200);
    fill(255,0,0);
    ellipse(x,y,radius,radius);

    var r = random(440,880);
    //osc0.freq(r);
    osc0.freq((x/900.0)*440 + 440);
    osc0.amp(y/600);
    }

    if (t>900)
    {
        osc0.stop();
    }

    //println(t);

    t += 1.0;
}
