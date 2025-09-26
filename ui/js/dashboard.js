// Dashboard data management
import { appState } from './state.js';

// Track progress data
let progressData = null;

/**
 * Fetches dashboard data based on the current topic
 * @param {string} topic - The current topic
 * @returns {Promise<Object>} - Dashboard data with popular topics and recent activities
 */
export async function fetchDashboardData(topic = "") {
  try {
    const response = await fetch('/dashboard_data', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-store',
      },
      body: JSON.stringify({ topic }),
      cache: 'no-store',
    });
    
    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }
    
    const data = await response.json();
    return {
      ok: true,
      data
    };
  } catch (error) {
    console.error('Error fetching dashboard data:', error);
    return {
      ok: false,
      error: error.message
    };
  }
}

/**
 * Updates the dashboard UI with the fetched data
 * @param {Object} dashboardData - The dashboard data
 */
export function updateDashboardUI(dashboardData) {
  if (!dashboardData || !dashboardData.ok) {
    console.error('Invalid dashboard data');
    return;
  }
  
  const data = dashboardData.data;
  
  // Update stats
  updateStats();
  
  // Update popular topics
  updatePopularTopics(data.popular_topics);
  
  // Update recent activities
  updateRecentActivities(data.recent_activities);
  
  // Update learning path
  updateLearningPath();
}

/**
 * Fetches progress data for the current session
 * @returns {Promise<Object>} - Progress data
 */
export async function fetchProgressData() {
  if (!appState.sessionId) return null;
  
  try {
    const response = await fetch(`/progress/${appState.sessionId}`, { cache: 'no-store', headers: { 'Cache-Control': 'no-store' } });
    
    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }
    
    const data = await response.json();
    progressData = data; // Store in module scope
    return data;
  } catch (error) {
    console.error('Error fetching progress data:', error);
    return null;
  }
}

/**
 * Updates the stats section with real progress data
 */
async function updateStats() {
  const statsCards = document.querySelectorAll('.stat-card .stat-value');
  if (!statsCards.length) return;
  
  // Try to get progress data if not already available
  if (!progressData) {
    progressData = await fetchProgressData();
  }
  
  if (progressData) {
    // Update stats with real data
    statsCards[0].textContent = '1'; // Sessions (always 1 for now)
    statsCards[1].textContent = progressData.topics_completed || '0';
    statsCards[2].textContent = `${progressData.accuracy || '0'}%`;
    
    // Add tooltip with additional mastery info
    const accuracyCard = statsCards[2].closest('.stat-card');
    if (accuracyCard) {
      accuracyCard.title = `Mastery Level: ${progressData.mastery_level}/10\nCurrent Difficulty: ${progressData.current_difficulty}\nStrategy: ${progressData.current_strategy}\nStreak: ${progressData.streak}`;
    }
  } else {
    // Fallback to placeholder data
    statsCards[0].textContent = '1';
    statsCards[1].textContent = '0';
    statsCards[2].textContent = '0%';
  }
}

/**
 * Updates the popular topics section
 * @param {Array<string>} topics - List of popular topics
 */
function updatePopularTopics(topics) {
  const topicTagsContainer = document.querySelector('.topic-tags');
  if (!topicTagsContainer) return;
  
  // Clear existing topics
  topicTagsContainer.innerHTML = '';
  
  if (!topics || topics.length === 0) {
    const emptyState = document.createElement('p');
    emptyState.className = 'empty-state';
    emptyState.textContent = 'No related topics available';
    topicTagsContainer.appendChild(emptyState);
    return;
  }
  
  // Add new topics
  topics.forEach(topic => {
    const topicTag = document.createElement('div');
    topicTag.className = 'topic-tag';
    topicTag.textContent = topic;
    
    // Add click event to set the topic in the input field
    topicTag.addEventListener('click', () => {
      const topicInput = document.getElementById('topic');
      if (topicInput) {
        topicInput.value = topic;
        topicInput.focus();
      }
    });
    
    topicTagsContainer.appendChild(topicTag);
  });
}

/**
 * Updates the recent activities section
 * @param {Array<Object>} activities - List of recent activities
 */
function updateRecentActivities(activities) {
  const activitiesContainer = document.querySelector('.card:has(.activity-item) .card-body');
  if (!activitiesContainer) return;
  
  // Clear existing activities
  activitiesContainer.innerHTML = '';
  
  if (!activities || activities.length === 0) {
    const emptyState = document.createElement('p');
    emptyState.className = 'empty-state';
    emptyState.textContent = 'No recent activities available';
    activitiesContainer.appendChild(emptyState);
    return;
  }
  
  // Add new activities
  activities.forEach(activity => {
    const activityItem = document.createElement('div');
    activityItem.className = 'activity-item';
    
    // Set icon based on activity type
    let iconClass = 'fa-book';
    if (activity.type === 'Completed') iconClass = 'fa-check';
    else if (activity.type === 'Mastered') iconClass = 'fa-star';
    
    activityItem.innerHTML = `
      <div class="activity-icon">
        <i class="fas ${iconClass}"></i>
      </div>
      <div class="activity-content">
        <div class="activity-title">${activity.type} "${activity.topic}"</div>
        <div class="activity-meta">${activity.time}</div>
      </div>
    `;
    
    activitiesContainer.appendChild(activityItem);
  });
}

/**
 * Updates the learning path section with mastery progression
 */
function updateLearningPath() {
  const pathContainer = document.querySelector('.card:has(.path-item) .card-body');
  if (!pathContainer) return;
  
  // Clear existing path
  pathContainer.innerHTML = '';
  
  // Create learning path structure
  const pathDiv = document.createElement('div');
  pathDiv.className = 'learning-path';
  
  const pathLine = document.createElement('div');
  pathLine.className = 'path-line';
  pathDiv.appendChild(pathLine);
  
  // Add current topic as the first item
  const currentTopic = document.getElementById('topic')?.value || 'Current Topic';
  
  // Calculate progress percentage based on mastery level
  let masteryLevel = 0;
  let progressPercent = 10; // Default starting progress
  let currentDifficulty = 'beginner';
  let currentStrategy = 'neural_compression';
  
  if (progressData) {
    masteryLevel = progressData.mastery_level || 0;
    progressPercent = Math.max(10, Math.min(100, masteryLevel * 10)); // 0-10 scale to 0-100%
    currentDifficulty = progressData.current_difficulty || 'beginner';
    currentStrategy = progressData.current_strategy || 'neural_compression';
  }
  
  // Format strategy name for display
  const strategyDisplay = currentStrategy
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
  
  const pathItem = document.createElement('div');
  pathItem.className = 'path-item';
  
  pathItem.innerHTML = `
    <div class="path-marker">1</div>
    <div class="path-content">
      <div class="path-title">${currentTopic}</div>
      <div class="path-meta">
        <span class="difficulty-badge ${currentDifficulty}">${currentDifficulty}</span>
        <span class="strategy-badge">${strategyDisplay}</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width: ${progressPercent}%;"></div>
      </div>
    </div>
  `;
  
  pathDiv.appendChild(pathItem);
  pathContainer.appendChild(pathDiv);
}

/**
 * Initializes the dashboard with data based on the current topic
 * @param {string} topic - The current topic
 */
export async function initDashboard(topic = "") {
  if (!topic) {
    console.warn('No topic provided for dashboard initialization');
    return;
  }
  
  try {
    // Fetch progress data first (if we have a session)
    if (appState.sessionId) {
      progressData = await fetchProgressData();
    }
    
    // Then fetch dashboard data
    const result = await fetchDashboardData(topic);
    if (result.ok) {
      updateDashboardUI(result);
    } else {
      console.error('Failed to fetch dashboard data:', result.error);
    }
  } catch (error) {
    console.error('Error initializing dashboard:', error);
  }
}