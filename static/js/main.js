/* SolarSense AI — global front-end behaviour */

document.addEventListener('DOMContentLoaded', () => {
  // Re-trigger fade-up animation on elements as they scroll into view
  const targets = document.querySelectorAll('.fade-up');
  if ('IntersectionObserver' in window && targets.length) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.animationPlayState = 'running';
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.15 });
    targets.forEach(el => io.observe(el));
  }

  // Auto-dismiss flash alerts after a few seconds
  document.querySelectorAll('.alert-solar').forEach(alertEl => {
    setTimeout(() => {
      alertEl.style.transition = 'opacity .4s';
      alertEl.style.opacity = '0';
      setTimeout(() => alertEl.remove(), 400);
    }, 6000);
  });
});
