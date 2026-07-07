function calculateScore() {

    let age = parseInt(document.getElementById("age").value);
    let income = parseInt(document.getElementById("income").value);
    let loan = parseInt(document.getElementById("loan").value);
    let debt = parseInt(document.getElementById("debt").value);

    let score = 300;

    if (age >= 21) score += 100;
    if (income >= 30000) score += 150;
    if (loan <= income * 2) score += 100;
    if (debt <= income * 0.5) score += 150;

    let status = "";

    if (score >= 700) {
        status = "Low Risk";
    } else if (score >= 500) {
        status = "Medium Risk";
    } else {
        status = "High Risk";
    }

    document.getElementById("result").innerHTML =
        "Credit Score: " + score + "<br>Risk Level: " + status;
}

