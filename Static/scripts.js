document.getElementById('predictor-form').addEventListener('submit', function(event) {
    const age = document.getElementById('age').value;
    const bmi = document.getElementById('bmi').value;
    const cycleLength = document.getElementById('cycle-length').value;

    if (age < 0 || bmi < 0 || cycleLength <= 0) {
        alert('Please enter valid positive numbers.');
        event.preventDefault();
    }
    function goBack() {
            window.history.back();
        }
});
