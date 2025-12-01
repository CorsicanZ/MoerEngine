// Runtime
#include "scene/Scene.h"
#include "config/ConfigManager.h"
#include "loader/LoaderInterface.h"
#include "log/LogSystem.h"
#include "rhi/RHI.h"
#include "taskgraph/TaskSystem.h"

// 3rd party (std)
#include <cassert>

// namespace
using namespace Moer;
using namespace Moer::Render;

int main(const int argc, const char** argv) {
    // Init LogSystem
    LogSystem::Init(); // for LOG_DEBUG & LOG_TRACE when debug mode

    // Init ConfigManager
    std::filesystem::path path = argv[0];
    path = path.filename().string().find(".exe") != std::string::npos ? path.parent_path() : path;

    ConfigManager::GetInstance().Init(path);

    // Init TaskSystem
    TaskSystem::Init();

    // Init RenderDevice
    auto     rhi_type_str = ConfigManager::GetInstance().GetConfig().engine.rhi.type;
    ERHIType rhi_type     = [&]() {
        if (rhi_type_str == "vulkan") {
            LOG_INFO("Using Vulkan as RHI backend");
            return ERHIType::Vulkan;
        }
        if (rhi_type_str == "d3d12") {
            LOG_INFO("Using D3D12 as RHI backend");
            return ERHIType::D3D12;
        }

        LOG_WARNING(
            "Unknown RHI type '{}', fallback to Vulkan",
            ConfigManager::GetInstance().GetConfig().engine.rhi.type
        );
        return ERHIType::Vulkan;
    }();
    RenderDevice::Init(std::move(DeviceInitInfo{
        .rhi_type        = rhi_type,
        .name            = "MoerEngine",
        .rhi_api_version = ConfigManager::GetInstance().GetConfig().engine.rhi.api_version,
    }));

    {
        // Get a lot of things
        auto&       device              = RenderDevice::Get();
        auto        bindless_array      = device.CreateBindlessArray();
        Scene       scene               = {bindless_array};
        auto&       gfx_queue           = device.GetCommandQueue(EQueueType::Graphics);
        auto&       copy_queue          = device.GetCopyQueue();
        auto        copy_queue_timeline = copy_queue.GetFenceHandle();
        CommandList cmd_list            = {};

        // MARK: Scene
        Resource::LoaderInterface::LoadSceneFromFileAsync(ConfigManager::GetInstance().GetScenePath(), scene);
        auto&& scope_exit_reset_async_load_info = OnScopeExit([&] {
            scene.ResetAsyncLoadInfo();
        });

        while (!scene.GetCurrentSceneLoadInfo().Get() || !scene.GetCurrentSceneLoadInfo()->IsReady()) {
        }
        scene.Info();
    }

    RenderDevice::Dispose();
    TaskSystem::ShutDown();

    return 0;
}