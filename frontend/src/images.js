// 모든 이미지는 frontend/public/ 에 위치 (로컬 파일)
export const RECIPE_IMAGES = {
  // 실제 존재하는 파일 (.png)
  "add-user":                    "/add-user.png",
  "air-fryer":                   "/air-fryer.png",
  blender:                       "/blender.png",
  "chef-mascot":                 "/chef-mascot.png",
  "citrus-juicer":               "/citrus-juicer.png",
  "coffe-machine":               "/coffe-machine.png",
  "cook-bg-brown":               "/cook-bg-brown.png",
  "cook-bg-yellow":              "/cook-bg-yellow.png",
  "cook-potato-wink":            "/cook-potato-wink.png",
  cooked:                        "/cooked.png",
  "exit-icon":                   "/exit-icon.png",
  "food-steamer":                "/food-steamer.png",
  house:                         "/house.png",
  "left-arrow":                  "/left-arrow.png",
  "login-naver":                 "/login-naver.png",
  "main-bg":                     "/main-bg.png",
  "main-character":              "/main-character.png",
  "main-next":                   "/main-next.png",
  "main-profile":                "/main-profile.png",
  "main-weather":                "/main-weather.png",
  "my-recipe-board":             "/my-recipe-board.png",
  "my-recipe-borderline-beige":  "/my-recipe-borderline-beige.png",
  "my-recipe-clip-beige":        "/my-recipe-clip-beige.png",
  "my-recipe-clip-orange":       "/my-recipe-clip-orange.png",
  "my-recipe-close":             "/my-recipe-close.png",
  "my-recipe-level":             "/my-recipe-level.png",
  "my-recipe-time":              "/my-recipe-time.png",
  "nav-chat-click":              "/nav-chat-click.png",
  "nav-chat-non":                "/nav-chat-non.png",
  "nav-cook-click":              "/nav-cook-click.png",
  "nav-cook-non":                "/nav-cook-non.png",
  "nav-home-non":                "/nav-home-non.png",
  "nav-my-click":                "/nav-my-click.png",
  "nav-my-non":                  "/nav-my-non.png",
  oven:                          "/oven.png",
  "potato-face":                 "/potato-face.png",
  potato_face:                   "/potato-face.png",
  "rice-cooker":                 "/rice-cooker.png",
  "splash-bg":                   "/splash-bg.png",
  "splash-potato":               "/splash-potato.png",
  "stovetop-waffle":             "/stovetop-waffle.png",
  "toast-appliance":             "/toast-appliance.png",

  // public/ 에 없는 키 → 유사한 기존 파일로 대체
  "back-icon":                   "/left-arrow.png",
  "cook-bg-green":               "/cook-bg-brown.png",
  "cook-complete-alert":         "/cook-potato-wink.png",
  "cook-peu-image":              "/chef-mascot.png",
  "level-icon":                  "/my-recipe-level.png",
  "time-icon":                   "/my-recipe-time.png",
  "nav-home-click":              "/nav-home-non.png",   // click 파일 없음 → non으로 대체
  "birthday-main-character_v2":  "/main-character.png", // 파일 없음 → main-character 대체

  // 날씨 아이콘 (파일 없음 → main-weather 대체)
  cloud:  "/main-weather.png",
  rain:   "/main-weather.png",
  snow:   "/main-weather.png",
  storm:  "/main-weather.png",
  sun:    "/main-weather.png",
  wind:   "/main-weather.png",

  // 그 외 기존 파일
  "loading-motion":   "/loading-motion.gif",
  "loading-bg-phone": "/loading-bg-phone.png",

  // 레시피 기본 이미지
  default_img: "/default-food.jpg",
};
