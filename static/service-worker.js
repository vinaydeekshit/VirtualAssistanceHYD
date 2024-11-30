self.addEventListener('install', event => {
    console.log('Service Worker installing.');
});

self.addEventListener('activate', event => {
    console.log('Service Worker activating.');
});

self.addEventListener('push', event => {
    const data = event.data ? event.data.json() : {};
    const title = 'Gate Status';
    const options = {
        body: data.gateStatus ? `${data.gateStatus}\nBoarding: ${data.boardingPercent}%` : 'No gate status available',
        icon: '/static/images/gmr.png'
    };

    event.waitUntil(
        self.registration.showNotification(title, options)
    );
});
