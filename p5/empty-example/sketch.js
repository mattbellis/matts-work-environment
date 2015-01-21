var t = 0.0;

function setup() {
  // put setup code here
    createCanvas(900, 600);    // **change** size() to createCanvas()
    stroke(255);               // stroke() is the same
    //noFill();                  // noFill() is the same
}

function draw() {
  // put drawing code here
    background(100);
    ellipse(50,50,20*sin(t/10.0),40);

    //println(t);

    t += 1.0;
}
