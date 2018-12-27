//import ddf.minim.*;
var mic;

var t = 0.0;

///////////////////////////////////////////////////////////////////////////////
// Info for particles
///////////////////////////////////////////////////////////////////////////////
var x = [1000][3];
var p = [1000][3];

var quote_timer = 0;
var quote_num = 0;




///////////////////////////////////////////////////////////////////////////////
function setup() {
  // put setup code here
    createCanvas(windowWidth, windowHeight);    // **change** size() to createCanvas()
    stroke(255);               // stroke() is the same
    //noFill();                  // noFill() is the same
    mic = new p5.AudioIn();
  mic.start();
}

///////////////////////////////////////////////////////////////////////////////
function draw() {

     /////////////////////////////////////////////////////////////////////////////
      // draw the waveforms
      /////////////////////////////////////////////////////////////////////////////
      var tot = mic.getLevel();
        if (tot>0.002) {
            print("tot: "+tot);
      }

  // put drawing code here
    background(0);

    micLevel = mic.getLevel();
    stroke(255);
    fill(255);
   ellipse(width/2, constrain(height-micLevel*height*5, 0, height), 10, 10);

    if (quote_timer==0) {
        quote = qft_quotes();
        print("here!");
    }

    fill(255, 255, 255,200-quote_timer);
    stroke(200-quote_timer);
    text(quote, 15, 30,600,400);
    textSize(24);

    quote_timer += 1;

    if (quote_timer>=200) {
        quote_timer=0;
    }



    t += 1.0;
}

///////////////////////////////////////////////////////////////////////////////
window.onresize = function() {
      canvas.size(windowWidth, windowHeight);
};

///////////////////////////////////////////////////////////////////////////////
// Quotes
///////////////////////////////////////////////////////////////////////////////
function qft_quotes()
{
    print("Quotes!");
    var s = "";
    var r = -1;
    while (r==quote_num || r==-1)
    {
        r = ceil(random(4));
        print("r: "+r);
    }

    if (r==1) {
        s = "For those who are not shocked when they first come across quantum theory cannot possibly have understood it.\n- Niels Bohr";
    } else if (r==2) {
        s = "I think I can safely say that nobody understands quantum mechanics.\n-Richard Feynman";
    } else if (r==3) {
        s = "I do not like it, and I am sorry I ever had anything to do with it.\n-Erwin Schrodinger";
    } else if (r==4) {
        s = "What nature demands from us is not a quantum theory or a wave theory; rather, nature demands from us a synthesis of these two views which thus far has exceeded the mental powers of physicists.\n-Albert Einstein";
    }

    quote_num=r;
    return s;

}

