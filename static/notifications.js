

document.addEventListener('DOMContentLoaded', function() {
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('static/service-worker.js').then(function(registration) {
            console.log('Service Worker registered with scope:', registration.scope);
        }).catch(function(error) {
            console.log('Service Worker registration failed:', error);
        });
    }

    const flightNumber = localStorage.getItem('flightNumber');
    if (!flightNumber) {
        console.log('No flight number available.');
        alert('No flight number available.');
        return;
    }

    function checkGateStatus() {
        fetch('/store_boarding_gate_status', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ flno: flightNumber })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.log('Error fetching gate status:', data.error);
                alert('Error fetching gate status.');
                return;
            }

            const gateStatus = data.status;
            const boardingPercent = data.boardingPercent;

            if (gateStatus) {
                let notificationBody = `Flight Number: ${flightNumber}\nStatus: ${gateStatus}`;

                if (gateStatus.toLowerCase() === 'gateopened' && boardingPercent) {
                    notificationBody += `\nBoarding: ${boardingPercent}%`;
                }

                const isMobile = /Mobi|Android|iPhone|iPad|iPod|BlackBerry|Windows Phone|Opera Mini|IEMobile|Mobile/i.test(navigator.userAgent);

                if (isMobile) {
                    alert(notificationBody);
                } else {
                    if (Notification.permission === 'granted') {
                        new Notification('Gate Status', {
                            body: notificationBody,
                            icon: '/static/images/gmr.png' 
                        });
                    } else if (Notification.permission === 'denied') {
                        alert(notificationBody);
                    } else {
                        Notification.requestPermission().then(permission => {
                            if (permission === 'granted') {
                                new Notification('Gate Status', {
                                    body: notificationBody,
                                    icon: '/static/images/gmr.png' 
                                });
                            } else {
                                alert(notificationBody);
                            }
                        });
                    }
                }

                if (gateStatus.toLowerCase() !== 'gateopened') {
                    clearInterval(intervalId);
                    console.log('Gate closed, stopping notifications.');
                }
            } else {
                console.log('No gate status available.');
                alert('No gate status available.');
            }
        })
        .catch(error => {
            console.log('Error:', error);
            alert('Error checking gate status.');
        });
    }

    
    checkGateStatus();
    var intervalId = setInterval(checkGateStatus, 5 * 60 * 1000);
});
