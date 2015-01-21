var t = 0.0;
var oscs = [new p5.SinOsc(), new p5.SinOsc(), new p5.SinOsc(), new p5.SinOsc()];
var freqs = [440,240,909,200];
var pan_start = [-1,1,-1,1];
var pan_step = [0.05,-0.01,0.02,-0.05];
var i = 0;

function setup() {
  // put setup code here
    createCanvas(900, 600);    // **change** size() to createCanvas()
    stroke(255);               // stroke() is the same
    for(i=0;i<oscs.length;i++)
    {
        oscs[i].amp(0.5);
    }
    //noFill();                  // noFill() is the same
}

function draw() {
  // put drawing code here
    background(200);
    fill(255,0,0);
    ellipse(50,50,20*sin(t/10.0),40);

    if (t<1)
    {
        for(i=0;i<oscs.length;i++)
        {
            oscs[i].start();
        }
    }
    
    for(i=0;i<oscs.length;i++)
    {
        oscs[i].freq(freqs[i]);
        oscs[i].pan(pan_start[i]+t*pan_step[i]);
        oscs[i].amp(0.5-0.005*t);
    }

    if (t>100)
    {
        for(i=0;i<oscs.length;i++)
        {
            oscs[i].stop();
        }
    }

    //println(t);

    t += 1.0;
}
