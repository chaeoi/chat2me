#include <stdio.h>
#include <string.h>

#include "bsp_axp2101.h"
#include "bsp_display.h"
#include "bsp_i2c.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_io_expander_tca9554.h"
#include "esp_lcd_panel_ops.h"
#include "esp_log.h"
#include "esp_system.h"
#include "logo_mark_data.h"
#include "lv_port.h"

#define EXAMPLE_DISPLAY_ROTATION LV_DISP_ROT_180
#define EXAMPLE_LCD_H_RES 320
#define EXAMPLE_LCD_V_RES 480
#define LCD_BUFFER_PIXELS (EXAMPLE_LCD_H_RES * EXAMPLE_LCD_V_RES)
#define LCD_TRANSFER_PIXELS (LCD_BUFFER_PIXELS / 10)
#define LCD_TRANSFER_BYTES (LCD_TRANSFER_PIXELS * sizeof(uint16_t))

#define BOOT_PATTERN_MS 120
#define BOOT_PATTERN_LINES 32
#define DISPLAY_LINE_SIZE 512
#define DISPLAY_BRIGHTNESS 100
#define THINKING_TIMEOUT_MS 15000
#define ANIMATION_FRAME_MS 33
#define STARTUP_REFRESH_COUNT 8
#define STARTUP_REFRESH_DELAY_MS 140
#define UI_INIT_RETRY_COUNT 60
#define UI_INIT_LOCK_TIMEOUT_MS 500
#define UI_INIT_RETRY_DELAY_MS 100

static const char *TAG = "chat2me_display";

static esp_io_expander_handle_t expander_handle = NULL;
static esp_lcd_panel_io_handle_t io_handle = NULL;
static esp_lcd_panel_handle_t panel_handle = NULL;
static lv_disp_t *lvgl_disp = NULL;

static lv_obj_t *root = NULL;
static bool ui_ready = false;
static bool backlight_enabled = false;

static char current_state[24] = "idle";
static uint32_t state_changed_ms = 0;

static lv_color_t color_bg = lv_color_hex(0x020607);

extern "C" void app_main(void);
void lv_port_init(void);
static void set_display_state_locked(const char *state);
static void apply_state_ui(void);
static void animate_cb(lv_timer_t *timer);
static void force_display_refresh_locked(void);

static void draw_boot_pattern(void)
{
    uint16_t *band = (uint16_t *)heap_caps_malloc(
        EXAMPLE_LCD_H_RES * BOOT_PATTERN_LINES * sizeof(uint16_t), MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    if (!band) {
        ESP_LOGW(TAG, "boot pattern band allocation failed");
        return;
    }

    ESP_LOGI(TAG, "clearing display");
    for (int y = 0; y < EXAMPLE_LCD_V_RES; y += BOOT_PATTERN_LINES) {
        int h = BOOT_PATTERN_LINES;
        if (y + h > EXAMPLE_LCD_V_RES) {
            h = EXAMPLE_LCD_V_RES - y;
        }

        for (int row = 0; row < h; ++row) {
            for (int x = 0; x < EXAMPLE_LCD_H_RES; ++x) {
                band[row * EXAMPLE_LCD_H_RES + x] = 0x0000;
            }
        }

        ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, y, EXAMPLE_LCD_H_RES, y + h, band));
    }

    vTaskDelay(pdMS_TO_TICKS(BOOT_PATTERN_MS));
    heap_caps_free(band);
}

static void io_expander_init(i2c_master_bus_handle_t bus_handle)
{
    ESP_ERROR_CHECK(esp_io_expander_new_i2c_tca9554(
        bus_handle, ESP_IO_EXPANDER_I2C_TCA9554_ADDRESS_000, &expander_handle));
    ESP_ERROR_CHECK(esp_io_expander_set_dir(expander_handle, IO_EXPANDER_PIN_NUM_1, IO_EXPANDER_OUTPUT));
    ESP_ERROR_CHECK(esp_io_expander_set_level(expander_handle, IO_EXPANDER_PIN_NUM_1, 0));
    vTaskDelay(pdMS_TO_TICKS(100));
    ESP_ERROR_CHECK(esp_io_expander_set_level(expander_handle, IO_EXPANDER_PIN_NUM_1, 1));
    vTaskDelay(pdMS_TO_TICKS(200));
}

void lv_port_init(void)
{
    lvgl_port_cfg_t port_cfg = {};
    port_cfg.task_priority = 4;
    port_cfg.task_stack = 1024 * 5;
    port_cfg.task_affinity = 1;
    port_cfg.task_max_sleep_ms = 500;
    port_cfg.timer_period_ms = 5;
    lvgl_port_init(&port_cfg);

    lvgl_port_display_cfg_t disp_cfg = {};
    disp_cfg.io_handle = io_handle;
    disp_cfg.panel_handle = panel_handle;
    disp_cfg.buffer_size = LCD_BUFFER_PIXELS;
    disp_cfg.sw_rotate = EXAMPLE_DISPLAY_ROTATION;
    disp_cfg.hres = EXAMPLE_LCD_H_RES;
    disp_cfg.vres = EXAMPLE_LCD_V_RES;
    disp_cfg.trans_size = LCD_TRANSFER_PIXELS;
    disp_cfg.draw_wait_cb = NULL;
    disp_cfg.flags.buff_dma = false;
    disp_cfg.flags.buff_spiram = true;

    if (disp_cfg.sw_rotate == LV_DISP_ROT_180 || disp_cfg.sw_rotate == LV_DISP_ROT_NONE) {
        disp_cfg.hres = EXAMPLE_LCD_H_RES;
        disp_cfg.vres = EXAMPLE_LCD_V_RES;
    } else {
        disp_cfg.hres = EXAMPLE_LCD_V_RES;
        disp_cfg.vres = EXAMPLE_LCD_H_RES;
    }
    lvgl_disp = lvgl_port_add_disp(&disp_cfg);
}

static int clamp_int(int value, int min_value, int max_value)
{
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static lv_obj_t *logo_img = NULL;
static lv_img_dsc_t logo_dsc = {};
static uint8_t *logo_pixels = NULL;
static int logo_base_x = 0;
static int logo_base_y = 0;
static int logo_current_zoom = 256;
static uint32_t last_anim_ms = 0;

static void fill_logo_pixels(void)
{
    uint16_t white = lv_color_white().full;
    for (int i = 0; i < LOGO_MARK_WIDTH * LOGO_MARK_HEIGHT; i++) {
        logo_pixels[i * LV_IMG_PX_SIZE_ALPHA_BYTE + 0] = (uint8_t)(white & 0xff);
        logo_pixels[i * LV_IMG_PX_SIZE_ALPHA_BYTE + 1] = (uint8_t)((white >> 8) & 0xff);
        logo_pixels[i * LV_IMG_PX_SIZE_ALPHA_BYTE + 2] = kLogoMarkAlpha[i];
    }
}

static int smooth_follow(int current, int target, int divisor)
{
    int delta = target - current;
    if (delta == 0) {
        return current;
    }
    int step = delta / divisor;
    if (step == 0) {
        step = delta > 0 ? 1 : -1;
    }
    return current + step;
}

static int wave_value(uint32_t now, uint32_t period_ms, int phase_deg, int amplitude)
{
    int angle = (int)((now * 360 / period_ms + phase_deg) % 360);
    return lv_trigo_sin((int16_t)angle) * amplitude / LV_TRIGO_SIN_MAX;
}

static int eased_cycle(uint32_t now, uint32_t period_ms, int phase_deg)
{
    int angle = (int)((now * 360 / period_ms + phase_deg) % 360);
    return (lv_trigo_sin((int16_t)angle) + LV_TRIGO_SIN_MAX) * 255 / (2 * LV_TRIGO_SIN_MAX);
}

static void render_logo_animation(uint32_t now)
{
    if (!logo_img) {
        return;
    }

    bool listening = strcmp(current_state, "listening") == 0;
    bool thinking = strcmp(current_state, "thinking") == 0;
    bool speaking = strcmp(current_state, "speaking") == 0;
    bool error = strcmp(current_state, "error") == 0;
    bool active = listening || thinking || speaking || error;

    uint32_t elapsed = last_anim_ms == 0 ? 50 : now - last_anim_ms;
    last_anim_ms = now;
    int follow = elapsed > 120 ? 4 : 8;

    int zoom = 256;

    if (listening) {
        zoom = 258 + wave_value(now, 3200, 20, 5);
    } else if (thinking) {
        zoom = 257 + wave_value(now, 4600, 0, 8) + wave_value(now, 7200, 120, 3);
    } else if (speaking) {
        int cycle = eased_cycle(now, 1650, 0);
        int breathe = cycle < 128 ? cycle : 255 - cycle;
        zoom = 257 + breathe * 16 / 128 + wave_value(now, 2600, 45, 3);
    } else if (error) {
        int pulse = eased_cycle(now, 1050, 0);
        int breathe = pulse < 128 ? pulse : 255 - pulse;
        zoom = 256 + breathe * 12 / 128;
    }

    logo_current_zoom = smooth_follow(logo_current_zoom, zoom, follow);

    int current_zoom = active ? clamp_int(logo_current_zoom, 250, 278) : 256;

    lv_obj_set_pos(logo_img, logo_base_x, logo_base_y);
    lv_img_set_angle(logo_img, 0);
    lv_img_set_zoom(logo_img, (uint16_t)current_zoom);
    lv_obj_set_style_img_opa(logo_img, LV_OPA_COVER, 0);
    lv_obj_invalidate(logo_img);
}

static bool create_logo_mark(int logo_x, int logo_y)
{
    size_t bytes = LV_IMG_PX_SIZE_ALPHA_BYTE * LOGO_MARK_WIDTH * LOGO_MARK_HEIGHT;
    logo_pixels = (uint8_t *)heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!logo_pixels) {
        logo_pixels = (uint8_t *)heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    if (!logo_pixels) {
        ESP_LOGE(TAG, "logo allocation failed");
        return false;
    }

    fill_logo_pixels();
    logo_dsc.header.always_zero = 0;
    logo_dsc.header.w = LOGO_MARK_WIDTH;
    logo_dsc.header.h = LOGO_MARK_HEIGHT;
    logo_dsc.header.cf = LV_IMG_CF_TRUE_COLOR_ALPHA;
    logo_dsc.data_size = bytes;
    logo_dsc.data = logo_pixels;

    logo_img = lv_img_create(root);
    lv_obj_clear_flag(logo_img, LV_OBJ_FLAG_CLICKABLE | LV_OBJ_FLAG_SCROLLABLE);
    lv_img_set_src(logo_img, &logo_dsc);
    lv_img_set_pivot(logo_img, LOGO_MARK_WIDTH / 2, LOGO_MARK_HEIGHT / 2);
    lv_img_set_antialias(logo_img, true);
    lv_img_set_angle(logo_img, 0);
    lv_img_set_zoom(logo_img, 256);
    lv_obj_set_style_img_opa(logo_img, LV_OPA_COVER, 0);
    lv_obj_set_pos(logo_img, logo_x, logo_y);
    return true;
}

static void build_ui(void)
{
    root = lv_scr_act();
    lv_obj_clear_flag(root, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_style_bg_opa(root, LV_OPA_COVER, 0);
    lv_obj_set_style_bg_color(root, color_bg, 0);
    lv_obj_set_style_bg_grad_dir(root, LV_GRAD_DIR_NONE, 0);

    int logo_x = (EXAMPLE_LCD_H_RES - LOGO_MARK_WIDTH) / 2;
    int logo_y = (EXAMPLE_LCD_V_RES - LOGO_MARK_HEIGHT) / 2;
    logo_base_x = logo_x;
    logo_base_y = logo_y;
    logo_current_zoom = 256;
    create_logo_mark(logo_x, logo_y);
}

static void ensure_ui_ready_locked(void)
{
    if (ui_ready) {
        return;
    }

    build_ui();
    set_display_state_locked("idle");
    apply_state_ui();
    lv_timer_create(animate_cb, ANIMATION_FRAME_MS, NULL);
    animate_cb(NULL);
    force_display_refresh_locked();
    ui_ready = true;
    ESP_LOGI(TAG, "ui ready");
}

static bool initialize_ui_with_retry(void)
{
    for (int attempt = 1; attempt <= UI_INIT_RETRY_COUNT; ++attempt) {
        if (lvgl_port_lock(UI_INIT_LOCK_TIMEOUT_MS)) {
            ensure_ui_ready_locked();
            force_display_refresh_locked();
            lvgl_port_unlock();
            return true;
        }

        ESP_LOGW(TAG, "ui init lock timeout, retry %d/%d", attempt, UI_INIT_RETRY_COUNT);
        vTaskDelay(pdMS_TO_TICKS(UI_INIT_RETRY_DELAY_MS));
    }

    return false;
}

static void set_display_state_locked(const char *state)
{
    strncpy(current_state, state, sizeof(current_state) - 1);
    current_state[sizeof(current_state) - 1] = '\0';
    state_changed_ms = lv_tick_get();
}

static void apply_state_ui(void)
{
    render_logo_animation(lv_tick_get());
}

static void animate_cb(lv_timer_t *timer)
{
    (void)timer;
    uint32_t now = lv_tick_get();
    bool thinking = strcmp(current_state, "thinking") == 0;

    if (thinking && now - state_changed_ms > THINKING_TIMEOUT_MS) {
        set_display_state_locked("idle");
        apply_state_ui();
    }

    render_logo_animation(now);
}

static void force_display_refresh_locked(void)
{
    if (!root || !lvgl_disp) {
        return;
    }
    lv_obj_invalidate(root);
    lv_refr_now(lvgl_disp);
}

static void startup_refresh_task(void *arg)
{
    (void)arg;
    for (int i = 0; i < STARTUP_REFRESH_COUNT; i++) {
        vTaskDelay(pdMS_TO_TICKS(STARTUP_REFRESH_DELAY_MS));
        if (lvgl_port_lock(250)) {
            esp_err_t err = esp_lcd_panel_disp_on_off(panel_handle, true);
            if (err != ESP_OK) {
                ESP_LOGW(TAG, "display on retry failed: %s", esp_err_to_name(err));
            }
            ensure_ui_ready_locked();
            apply_state_ui();
            animate_cb(NULL);
            force_display_refresh_locked();
            lvgl_port_unlock();
            if (!backlight_enabled) {
                bsp_display_set_brightness(DISPLAY_BRIGHTNESS);
                backlight_enabled = true;
            }
        }
    }
    vTaskDelete(NULL);
}

// 极简 JSON 提取：仅支持 "key":"value" 形式的字符串字段，value 内不能含转义引号。
// 数字/布尔字段（如 ts）无法读取；需要更多字段时请改用真正的 JSON 解析。
static bool extract_json_value(const char *line, const char *key, char *out, size_t out_size)
{
    char needle[32];
    snprintf(needle, sizeof(needle), "\"%s\":\"", key);
    const char *start = strstr(line, needle);
    if (!start) {
        return false;
    }
    start += strlen(needle);
    const char *end = strchr(start, '"');
    if (!end) {
        return false;
    }
    size_t len = end - start;
    if (len >= out_size) {
        len = out_size - 1;
    }
    memcpy(out, start, len);
    out[len] = '\0';
    return true;
}

static void handle_line(const char *line)
{
    char state[24] = "";
    if (!extract_json_value(line, "state", state, sizeof(state))) {
        return;
    }

    if (lvgl_port_lock(100)) {
        ensure_ui_ready_locked();
        set_display_state_locked(state);
        apply_state_ui();
        force_display_refresh_locked();
        lvgl_port_unlock();
    }
}

static void uart_task(void *arg)
{
    // 协议为每行一条 JSON。超过 DISPLAY_LINE_SIZE 的行会被 fgets 拆成两段而解析失败；
    // 发送端 text 受 DISPLAY_TEXT_MAX_CHARS 限制，正常不会超长。
    while (true) {
        char line[DISPLAY_LINE_SIZE] = {};
        if (fgets(line, sizeof(line), stdin) != NULL) {
            handle_line(line);
        } else {
            vTaskDelay(pdMS_TO_TICKS(50));
        }
    }
}

extern "C" void app_main(void)
{
    bsp_display_backlight_force_off();

    i2c_master_bus_handle_t i2c_bus_handle = bsp_i2c_init();

    ESP_ERROR_CHECK(bsp_axp2101_init(i2c_bus_handle));
    io_expander_init(i2c_bus_handle);
    bsp_display_brightness_init();
    bsp_display_set_brightness(0);
    bsp_display_init(&io_handle, &panel_handle, LCD_TRANSFER_BYTES);
    draw_boot_pattern();

    lv_port_init();
    ESP_LOGI(TAG, "serial status input ready on console stdin");

    if (!initialize_ui_with_retry()) {
        ESP_LOGE(TAG, "ui init failed after retries, restarting");
        esp_restart();
    }
    bsp_display_set_brightness(DISPLAY_BRIGHTNESS);
    backlight_enabled = true;

    xTaskCreate(startup_refresh_task, "display_startup_refresh", 4096, NULL, 5, NULL);
    xTaskCreate(uart_task, "uart_status", 4096, NULL, 8, NULL);
}
