// This script prevents color flashing during page navigation
(function() {
  // Apply theme colors immediately before page content loads
  function applyThemeImmediately() {
    // Check if user prefers dark mode
    const prefersDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    // Get stored theme preference if available
    const storedTheme = localStorage.getItem('sphinx-theme');
    
    // Set theme attribute on html element
    if (storedTheme) {
      document.documentElement.setAttribute('data-theme', storedTheme);
    } else {
      document.documentElement.setAttribute('data-theme', prefersDarkMode ? 'dark' : 'light');
    }
    
    // Add class to body to prevent transition flashes
    document.body.classList.add('no-transitions');
    
    // Remove the class after page has loaded
    window.addEventListener('load', function() {
      setTimeout(function() {
        document.body.classList.remove('no-transitions');
      }, 300);
    });
  }
  
  // Run immediately
  applyThemeImmediately();
  
  // Also run when DOM content is loaded
  document.addEventListener('DOMContentLoaded', applyThemeImmediately);
})();
