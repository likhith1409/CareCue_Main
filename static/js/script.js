/*=============== SHOW MENU ===============*/
const showMenu = (toggleId, navId) =>{
    const toggle = document.getElementById(toggleId),
    nav = document.getElementById(navId)
    
    // Validate that variables exist
    if(toggle && nav){
        toggle.addEventListener('click', ()=>{
            // We add the show-menu class to the div tag with the nav__menu class
            nav.classList.toggle('show-menu')
        })
    }
}
showMenu('nav-toggle','nav-menu')

/*=============== When the user scrolls the page, execute myFunction==============*/
window.onscroll = function() {myFunction()};
            
function myFunction() {
  var winScroll = document.body.scrollTop || document.documentElement.scrollTop;
  var height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
  var scrolled = (winScroll / height) * 100;
  document.getElementById("myBar").style.width = scrolled + "%";
}

/*=============== REMOVE MENU MOBILE ===============*/
const navLink = document.querySelectorAll('.nav__link')

function linkAction(){
    const navMenu = document.getElementById('nav-menu')
    // When we click on each nav__link, we remove the show-menu class
    navMenu.classList.remove('show-menu')
}
navLink.forEach(n => n.addEventListener('click', linkAction))

/*=============== SCROLL SECTIONS ACTIVE LINK ===============*/
const sections = document.querySelectorAll('section[id]')

function scrollActive(){
    const scrollY = window.pageYOffset

    sections.forEach(current =>{
        const sectionHeight = current.offsetHeight,
              sectionTop = current.offsetTop - 50,
              sectionId = current.getAttribute('id')

        if(scrollY > sectionTop && scrollY <= sectionTop + sectionHeight){
            document.querySelector('.nav__menu a[href*=' + sectionId + ']').classList.add('active-link')
        }else{
            document.querySelector('.nav__menu a[href*=' + sectionId + ']').classList.remove('active-link')
        }
    })
}
window.addEventListener('scroll', scrollActive)

/*=============== CHANGE BACKGROUND HEADER ===============*/
function scrollHeader(){
    const nav = document.getElementById('header')
    // When the scroll is greater than 80 viewport height, add the scroll-header class to the header tag
    if(this.scrollY >= 80) nav.classList.add('scroll-header'); else nav.classList.remove('scroll-header')
}
window.addEventListener('scroll', scrollHeader)

/*=============== SHOW SCROLL UP ===============*/
function scrollUp(){
    const scrollUp = document.getElementById('scroll-up');
    // When the scroll is higher than 560 viewport height, add the show-scroll class to the a tag with the scroll-top class
    if(this.scrollY >= 560) scrollUp.classList.add('show-scroll'); else scrollUp.classList.remove('show-scroll')
}
window.addEventListener('scroll', scrollUp)

/*=============== DARK LIGHT THEME ===============*/
const themeButton = document.getElementById('theme-button')
const darkTheme = 'dark-theme'
const iconTheme = 'bx-toggle-right'

// Previously selected topic (if user selected)
const selectedTheme = localStorage.getItem('selected-theme')
const selectedIcon = localStorage.getItem('selected-icon')

// We obtain the current theme that the interface has by validating the dark-theme class
const getCurrentTheme = () => document.body.classList.contains(darkTheme) ? 'dark' : 'light'
const getCurrentIcon = () => themeButton.classList.contains(iconTheme) ? 'bx-toggle-left' : 'bx-toggle-right'

// We validate if the user previously chose a topic
if (selectedTheme) {
  // If the validation is fulfilled, we ask what the issue was to know if we activated or deactivated the dark
  document.body.classList[selectedTheme === 'dark' ? 'add' : 'remove'](darkTheme)
  themeButton.classList[selectedIcon === 'bx-toggle-left' ? 'add' : 'remove'](iconTheme)
}

// Activate / deactivate the theme manually with the button
themeButton.addEventListener('click', () => {
    // Add or remove the dark / icon theme
    document.body.classList.toggle(darkTheme)
    themeButton.classList.toggle(iconTheme)
    // We save the theme and the current icon that the user chose
    localStorage.setItem('selected-theme', getCurrentTheme())
    localStorage.setItem('selected-icon', getCurrentIcon())
})





var dropdowns = document.getElementsByClassName("dropdown-btn");
var i;

for (i = 0; i < dropdowns.length; i++) {
  dropdowns[i].addEventListener("click", function() {
    var content = this.nextElementSibling;
    if (content.style.maxHeight) {
      content.style.maxHeight = null;
    } else {
      // Adjust the value below to the maximum height you want for the dropdown
      content.style.maxHeight = content.scrollHeight + "px";
    }
  });
}


//chatbot


// voice assistent

function openHtmlPage(pageUrl) {
    window.location.href = pageUrl;
}
  
function getMedicineDetails() {
    var medicineName = document.getElementById("medicineName").value;
    fetch(`/get_details?name=${medicineName}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById("uses").innerText = data.uses.join(', ');
            document.getElementById("sideEffects").innerText = data.side_effects.join(', ');
        })
        .catch(error => console.error('Error:', error));
}

function expandContent() {
    var expandedContent = document.getElementById('expandedContent');
    var expandDiv = document.querySelector('.expand-content');

    // Toggle the visibility of the expanded content
    if (expandedContent.style.display === 'none') {
        expandedContent.style.display = 'block';
        expandDiv.style.display = 'block';
    } else {
        expandedContent.style.display = 'none';
        expandDiv.style.display = 'none';
    }
}




function displayMessage(role, message) {
    const chatbox = document.getElementById('chatbox');
    const messageElement = document.createElement('p');
    messageElement.textContent = message;
    messageElement.classList.add('chat-message', role === 'user' ? 'user-message' : 'bot-message');
    chatbox.appendChild(messageElement);
}

function sendMessage() {
    const userInput = document.getElementById('userInput').value;
    displayMessage('user', userInput);

    // Send user input to the server and get bot response
    fetch('/send_message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `userInput=${encodeURIComponent(userInput)}`,
    })
    .then(response => response.json())
    .then(data => {
        const botResponse = data.bot;
        displayMessage('bot', botResponse);
    })
    .catch(error => console.error('Error sending message:', error));
}

const chatbox = document.getElementById('chatbox');
const buttons = document.querySelectorAll('.chart-buttons button');

buttons.forEach(button => {
    button.addEventListener('click', function () {
        const mode = this.textContent.trim();
        chatbox.innerHTML = `<p>You are in ${mode.toLowerCase()} mode.</p>`;
    });
});


function closeModals() {
    document.getElementById('loginModal').style.display = 'none';
    document.getElementById('signupModal').style.display = 'none';
}


function openSignup() {
    document.getElementById('loginModal').style.display = 'none';
    document.getElementById('signupModal').style.display = 'block';
}

function openLogin() {
    document.getElementById('signupModal').style.display = 'none';
    document.getElementById('loginModal').style.display = 'block';
}

function closeLoginModal() {
    document.getElementById('loginModal').style.display = 'none';
}

function closeSignupModal() {
    document.getElementById('signupModal').style.display = 'none';
}


// Initialize a variable to track user login status
var userLoggedIn = false;

function toggleLoginSignup() {
    var loginModal = document.getElementById("loginModal");
    var signupModal = document.getElementById("signupModal");
    var userDropdown = document.getElementById("userDropdown");

    // Toggle login/signup form visibility
    loginModal.style.display = (loginModal.style.display === "block") ? "none" : "block";
    signupModal.style.display = "none"; // Hide signup form when toggling

    // Toggle user information dropdown visibility
    userDropdown.style.display = (userDropdown.style.display === "block") ? "none" : "block";

    // Check if user is logged in and adjust login/signup button visibility
    if (userLoggedIn) {
        document.getElementById("loginSignupButton").style.display = "none"; // Hide login/signup button
    } else {
        document.getElementById("loginSignupButton").style.display = "block"; // Show login/signup button
    }
}

function login() {
    var email = document.getElementById("email").value;
    var password = document.getElementById("password").value;

    // Make an AJAX request to your Flask backend to check login credentials
    fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            email: email,
            password: password,
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById("loginError").innerText = data.error;
        } else {
            // After successful login, update the UI and set userLoggedIn to true
            userLoggedIn = true;
            updateUIOnLogin(data.name);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });

    // Prevent form submission
    return false;
}

function signup() {
    var name = document.getElementById("signupName").value;
    var email = document.getElementById("signupEmail").value;
    var password = document.getElementById("signupPassword").value;
    var confirmPassword = document.getElementById("confirmPassword").value;

    // Add client-side validation to ensure password and confirmPassword match
    if (password !== confirmPassword) {
        document.getElementById("signupError").innerText = "Passwords do not match";
        return false;
    }

    // Make an AJAX request to your Flask backend to handle signup
    fetch('/signup', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            name: name,
            email: email,
            password: password,
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById("signupError").innerText = data.error;
        } else {
            // After successful signup, update the UI and set userLoggedIn to true
            userLoggedIn = true;
            updateUIOnLogin(data.name);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });

    // Prevent form submission
    return false;
}

// ... (your existing JavaScript code)




function logout() {
    // Make an AJAX request to your Flask backend to handle logout
    // Replace the following lines with your actual AJAX implementation
    // Use the /logout route in your Flask app
    // Handle response and display appropriate messages
    // For simplicity, I'm assuming you have a /logout route in your Flask app

    // Example AJAX request using fetch API
    fetch('/logout')
    .then(response => response.json())
    .then(data => {
        // After successful logout, update the UI and set userLoggedIn to false
        userLoggedIn = false;
        updateUIOnLogout();
    })
    .catch(error => {
        console.error('Error:', error);
    });

    // Prevent form submission
    return false;
}

// Update the updateUIOnLogin function
function updateUIOnLogin(userData) {
    var loginModal = document.getElementById("loginModal");
    var userLabel = document.getElementById("userLabel");
    var userOptions = document.querySelector(".nav__user-options");
    var loginSignupButton = document.getElementById("loginSignupButton");

    // Hide login/signup form
    loginModal.style.display = "none";

    // Show user information in the navbar
    userLabel.innerText = userData.email;
    userLabel.style.display = "inline-block";

    // Show user options dropdown
    userOptions.style.display = "block";

    // Hide login/signup button
    loginSignupButton.style.display = "none";
}

// Update the updateUIOnLogout function
function updateUIOnLogout() {
    var userLabel = document.getElementById("userLabel");
    var userOptions = document.querySelector(".nav__user-options");
    var loginSignupButton = document.getElementById("loginSignupButton");
    var userDropdown = document.getElementById("userDropdown");

    // Clear the user's email
    userLabel.innerText = '';
    userLabel.style.display = "none";

    // Hide user options dropdown
    userOptions.style.display = "none";

    // Show the login/signup button
    loginSignupButton.style.display = "block";

    // Hide user dropdown
    userDropdown.style.display = "none";
}


function updateUIOnLogin(userName) {
    var loginModal = document.getElementById("loginModal");
    var userLabel = document.getElementById("userLabel");
    var userOptions = document.querySelector(".nav__user-options");
    var loginSignupButton = document.getElementById("loginSignupButton");

    // Hide login/signup form
    loginModal.style.display = "none";

    // Show user information in the navbar
    userLabel.innerText = userName;
    userLabel.style.display = "inline-block";

    // Show user options dropdown
    userOptions.style.display = "block";

    // Hide login/signup button
    loginSignupButton.style.display = "none";
}



// JavaScript for closing the login modal
function closeLogin() {
    document.getElementById('loginModal').style.display = 'none';
}

// JavaScript for closing the signup modal
function closeSignup() {
    document.getElementById('signupModal').style.display = 'none';
}

// JavaScript for closing modals when clicking outside the modal content
window.onclick = function(event) {
    if (event.target.className === 'modal') {
        event.target.style.display = 'none';
    }
};


