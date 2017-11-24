function setup() {
  // put setup code here
    createCanvas(710, 400, WEBGL);
    stroke(255);               // stroke() is the same
    //noFill();                  // noFill() is the same
}

function draw() {
  // put drawing code here
    background(100);
    //noStroke();

    push();
    translate(-100, 100);
    rotateY(1.25);
    rotateX(-0.9);
    box(100);
    pop();

    noFill();
    stroke(255);
    push();
    translate(500, height*0.35, -200);
    sphere(100);
    pop();

}
