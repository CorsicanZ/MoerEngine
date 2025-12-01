#pragma once

#include "math/Transform.h"

namespace Moer::ECS {
struct TransformComponent {
    Moer::Transform transform;

    // void Serialize(wi::Archive& archive, wi::ecs::EntitySerializer& seri);
};
} // namespace Moer::ECS