#pragma once
#include "Camera.h"
#include "serialize/Serializer.h"

namespace Moer::ECS {
struct CameraComponent {
    CameraRef camera;
    // void Serialize(wi::Archive& archive, wi::ecs::EntitySerializer& seri);
};

} // namespace Moer::ECS